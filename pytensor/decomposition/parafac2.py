from abc import abstractmethod
from pathlib import Path
import numpy as np
from .base_decomposer import BaseDecomposer
from . import decompositions
from . import cp
from ..utils import normalize_factors, get_pca_loadings
from .. import base


class BaseParafac2(BaseDecomposer):
    """
    TODO: Ikke tillat at evolve_mode og evolve_over settes manuelt
    TODO: Evolve_mode=2, evolve_over=0
    """
    DecompositionType = decompositions.Parafac2Tensor
    def __init__(self, 
        rank, 
        max_its, 
        convergence_tol=1e-10, 
        init='random',
        loggers=None,
        checkpoint_frequency=None,
        checkpoint_path=None,
    ):
        super().__init__(loggers=loggers, checkpoint_frequency=checkpoint_frequency, checkpoint_path=checkpoint_path)

        self.rank = rank
        self.max_its = max_its
        self.convergence_tol = convergence_tol
        self.init = init

    def set_target(self, X):
        if not isinstance(X, list):
            self.target_tensor = X
            X = X.transpose(2, 0, 1)
        
        self.X = X
        self.X_shape = [len(X[0]), [Xk.shape[1] for Xk in X], len(X)]    # len(A), len(Bk), len(C)
        self.X_norm = np.sqrt(sum(np.linalg.norm(Xk)**2 for Xk in X))
        self.num_X_elements = sum([np.prod(s) for s in self.X_shape])

    def init_random(self):
        """Random initialisation of the factor matrices
        """
        self.decomposition = self.DecompositionType.random_init(self.X_shape, rank=self.rank)

    def init_svd(self):
        """SVD initalisation
        """
        # TODO: This does not work for irregular tensors
        K = self.X_shape[2]
        J = self.X_shape[1]
        Y = np.zeros([J, J])

        for k in range(K):
            Y += self.X[k].T @ self.X[k]

        A = get_pca_loadings(Y, self.rank)
        blueprint_B = np.identity(self.rank)
        C = np.ones((K, self.rank))

        P = np.eye(J, self.rank)
        self.decomposition = self.DecompositionType(A, blueprint_B, C, P)

        self._update_projection_matrices()

    def init_components(self, initial_decomposition=None):
        if self.init.lower() == 'svd':
            self.init_svd()
        elif self.init.lower() == 'random':
            self.init_random()
        elif self.init.lower() == 'from_checkpoint':
            self.load_checkpoint(initial_decomposition)
        elif self.init.lower() == 'precomputed':
            self._check_valid_components(initial_decomposition)
            self.decomposition = initial_decomposition
        elif Path(self.init).is_file():
            self.load_checkpoint(self.init)
        else:
            # TODO: better message
            raise ValueError('Init method must be either `random`, `svd`, `from_checkpoint` or `precomputed`.')

    def _check_valid_components(self, decomposition):
        for i, factor_matrix, factor_name in zip([0, 2], [decomposition.A, decomposition.C], ['A', 'C']):
            if factor_matrix.shape[0] != self.X_shape[i]:
                raise ValueError(
                    f"The length of factor matrix {factor_name} ({factor_matrix.shape[0]}"
                    f"is not the same as the length of X's dimension {i} ({self.X_shape[i]})"
                )
            if factor_matrix.shape[1] != self.rank:
                raise ValueError(
                    f"The number of columns of {factor_name} ({factor_matrix.shape[1]}) does not agree with the models rank ({self.rank})"
                )
        
        for k, (B, X_slice) in enumerate(zip(decomposition.B, self.X)):
            if B.shape[0] != X_slice.shape[1]:
                raise ValueError(
                    f"The number of rows of factor matrix B_{k} ({B.shape[0]}"
                    f"is not the same as the number of columns of X_{k} ({X_slice.shape[1]})"
                )

            if B.shape[1] != self.rank:
                raise ValueError(
                    f"The number of columns of B_{k} ({B.shape[1]}) does not agree with the models rank ({self.rank})"
                )

    @abstractmethod
    def _fit(self):
        pass

    def fit(self, X, y=None, max_its=None, initial_decomposition=None):
        """Fit a parafac2 model. Precomputed components must be specified if init method is 'precomputed'
        """
        # TODO: initial_decomposition en parafc2tensor?
        # TODO: docstring

        self._init_fit(X=X, max_its=max_its, initial_decomposition=initial_decomposition)
        self._fit()

    def fit_transform(self, X, y=None, max_its=None, initial_decomposition=None):
        self.fit(X, y=None, max_its=None, initial_decomposition=None)
        return self.decomposition

    @property
    def loss(self):
        return self.SSE

    def _fit(self):
        return 1 - self.SSE/(self.X_norm**2)

    @property
    def SSE(self):
        SSE = 0
        for X_k, reconstructed_X_k, in zip(self.X, self.reconstructed_X):
            SSE += np.sum((X_k - reconstructed_X_k)**2)
        return SSE

    @property
    def MSE(self):
        return self.SSE/self.decomposition.num_elements
        
    @property
    def reconstructed_X(self):
        return self.decomposition.construct_slices()
    
    @property
    def projected_X(self):
        I = self.decomposition.A.shape[0]
        K = self.decomposition.C.shape[0]
        projected_X = np.empty((I, self.rank, K))

        for k, projection_matrix in enumerate(self.decomposition.projection_matrices):
            projected_X[..., k] = self.X[k]@projection_matrix
        return projected_X

    # Todo: Change name of this function
    def _update_projection_matrices(self):
        K = self.X_shape[2]

        for k in range(K):
            A = self.decomposition.A
            C = self.decomposition.C
            blueprint_B = self.decomposition.blueprint_B

            U, S, Vh = np.linalg.svd(self.X[k].T@((C[k]*A)@blueprint_B.T), full_matrices=False)
            S_tol = max(U.shape) * S[0] * (1e-16)
            # C[:, k]*A is equivalent to A@np.diag(C[:, k])
            should_keep = np.diag(S > S_tol).astype(float)

            self.decomposition.projection_matrices[k][...] = (Vh.T @ should_keep @ U.T).T

"""
            U, S, Vh = np.linalg.svd(
                self.decomposition.blueprint_B @ self.decomposition.D[..., k] \
                @ (self.decomposition.A.T @ self.X[k].T), full_matrices=False
            )

            S_tol = max(U.shape) * S[0] * (1e-16)
            should_keep = np.diag(S > S_tol).astype(float)

            self.decomposition.projection_matrices[k][...] = Vh.T @ should_keep @ U.T
"""            
            # Should_keep = diag([1, 1, ..., 1, 0, 0, ..., 0]) -> the zeros correspond to small singular values
            # Following Rasmus Bro's PARAFAC2 MATLAB script, which sets P_k = Q_k(Q_k'Q_k)^(-0.5) (line 524)
            #      Where the power is done by truncating very small singular values (for numerical stability)


class Parafac2_ALS(BaseParafac2):
    def __init__(
        self,
        rank,
        max_its,
        convergence_tol=1e-10,
        init='random',
        loggers=None,
        checkpoint_frequency=None,
        checkpoint_path=None,
        non_negativity_constraints=None,
        print_frequency=10,
        cp_updates_per_it=5,
    ):
        super().__init__(
            rank,
            max_its,
            convergence_tol=convergence_tol,
            init=init,
            loggers=loggers,
            checkpoint_frequency=checkpoint_frequency,
            checkpoint_path=checkpoint_path
        )
        self.non_negativity_constraints = non_negativity_constraints
        self.print_frequency = print_frequency
        self.cp_updates_per_it = cp_updates_per_it

    def _init_fit(self, X, max_its, initial_decomposition):
        super()._init_fit(X=X, max_its=max_its, initial_decomposition=initial_decomposition)
        self.prev_SSE = self.SSE
        self._rel_function_change = np.inf

    def _prepare_cp_decomposer(self):
        self.cp_decomposition = decompositions.KruskalTensor([self.decomposition.A, self.decomposition.blueprint_B, self.decomposition.C])
        self.cp_decomposer = cp.CP_ALS(self.rank, max_its=1000, 
                                       convergence_tol=0, print_frequency=-1, 
                                       non_negativity_constraints=self.non_negativity_constraints,
                                       init='precomputed')
        self.cp_decomposer._init_fit(X=self.projected_X, max_its=np.inf, initial_decomposition=self.cp_decomposition)

    def _fit(self):
        if self.non_negativity_constraints is None:
            self.non_negativity_constraints = [False, False, False]

        self._prepare_cp_decomposer()
        for it in range(self.max_its - self.current_iteration):
            if abs(self._rel_function_change) < self.convergence_tol:
                break

            self._update_parafac2_factors()
            self._update_convergence()

            if self.current_iteration % self.print_frequency == 0 and self.print_frequency > 0:
                print(f'{self.current_iteration:6d}: The MSE is {self.MSE:4g}, f is {self.loss:4g}, '
                      f'improvement is {self._rel_function_change:g}')

            self._after_fit_iteration()


        if (
            ((self.current_iteration+1) % self.checkpoint_frequency != 0) and 
            (self.checkpoint_frequency > 0)
        ):
            self.store_checkpoint()

    def _update_convergence(self):
        SSE = self.SSE
        self._rel_function_change = (self.prev_SSE - SSE)/self.prev_SSE
        self.prev_SSE = SSE

    def _update_parafac2_factors(self):
        #print('Before projection update') 
        #print(f'The MSE is {self.MSE: 4f}, f is {self.loss:4f}')
        self._update_projection_matrices()

        #print('Before ALS update') 
        #print(f'The MSE is {self.MSE: 4f}, f is {self.loss:4f}')

        self.cp_decomposer.set_target(self.projected_X)
        for _ in range(self.cp_updates_per_it):
            self.cp_decomposer._update_als_factors()
        self.decomposition.blueprint_B[...] *= self.cp_decomposer.weights
        self.cp_decomposition.weights = self.cp_decomposition.weights*0 + 1
        #print('After iteration') 
        #print(f'The MSE is {self.MSE: 4f}, f is {self.loss:4f}')
        # from pdb import set_trace; set_trace()

