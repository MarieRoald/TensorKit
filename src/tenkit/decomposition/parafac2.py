import warnings
from abc import abstractmethod
from pathlib import Path

import numpy as np

from .. import base, utils
from ..utils import get_pca_loadings, normalize_factors
from . import cp, decompositions
from .base_decomposer import BaseDecomposer

__all__ = ['Parafac2_ALS']


class BaseParafac2(BaseDecomposer):
    r"""Base class for Parafac2 decomposer objects

    Arguments:
    ----------
    rank: int
        Number of components 
    max_its: int (optional, default=1000)
        Maximum number of iterations for fitting the model. 
        Can be overwritten by the ``fit`` method.
    convergence_tol: float (optional, default=1e-6)
        Minimum relative function change between two consequetive
        iterations for the model to continue fitting. Computed as
        
        .. math::

            \frac{L(X_i) - L(X_{i+1})}{L(X_{i+1})},
        
        where :math:`L` is the loss (Sum squared error) and :math:`X_i`
        is the decomposition at iteration :math:`i`.
    init: str (optional, default = 'random')
        Initialisation scheme for the decomposer (case insensitive). Options:
    
          * Random: Initiate the decomposition as a random Parafac2 tensor
          * SVD: Use the SVD to find the factor matrices
          * CP: Run 20 CP iterations and use that decomposition as initial
          Parafac2 tensor (QR on the evolving mode to split into projection
          and blueprint matrices)
          * from_checkpoint: Load the last iteration in the specified
          checkpoint file (alternatively, just the path to the decomposition file)
          * Precomputed: Use already existing Parafac2 tensor to initialise
          the decomposition. 

    logger: list(Logger) (optional, default=None)
        List of loggers, each logger should implement a ``log`` method
        that takes a decomposer as input and a ``write_to_hdf5_group``
        method that stores the log in a hdf5 group. See 
        ``tenkit.logging.BaseLogger`` for interface.
    checkpoint_frequency: int (optional, default=None)
        How often the decomposer should store the decomposition and
        logs to disk. If None or negative, will only
        checkpoint the last iteration. 
    checkpoint_path: str or Path (optional, default=None)
        Where to store the log HDF5 file. If None, then the checkpoints
        and logs are not stored to disk.
    print_frequency: int (optional, default=None)
        How often convergence information should be printed in the terminal.
        None and negative values leads to no printing.
    """
    DecompositionType = decompositions.Parafac2Tensor
    def __init__(self, 
        rank, 
        max_its=1000, 
        convergence_tol=1e-6, 
        init='random',
        loggers=None,
        checkpoint_frequency=None,
        checkpoint_path=None,
        print_frequency=10,
    ):
        super().__init__(
            max_its=max_its,
            convergence_tol=convergence_tol,
            loggers=loggers,
            checkpoint_frequency=checkpoint_frequency,
            checkpoint_path=checkpoint_path,
            print_frequency=print_frequency,
        )

        self.rank = rank
        self.init = init

    def set_target(self, X):
        if not isinstance(X, list):
            self.target_tensor = X
            X = np.ascontiguousarray(X.transpose(2, 0, 1))
        
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

        P = [np.eye(J, self.rank) for k in range(K)]
        self.decomposition = self.DecompositionType(A, blueprint_B, C, P)

        self._update_projection_matrices()
    
    def init_cp(self):
        """CP initialisation. Input must be a tensor.
        """
        X = np.asarray(self.X)
        cp_als = cp.CP_ALS(self.rank, 20)
        cp_als.fit(X)
        C, A, B = cp_als.factor_matrices
        P, blueprint_B = np.linalg.qr(B)
        P = [P for _ in range(len(self.X))]
        self.decomposition = self.DecompositionType(A, blueprint_B, C, P)

    def init_components(self, initial_decomposition=None):
        if self.init.lower() == 'svd':
            self.init_svd()
        elif self.init.lower() == 'random':
            self.init_random()
        elif self.init.lower() == 'cp':
            self.init_cp()
        elif self.init.lower() == 'from_checkpoint':
            self.load_checkpoint(initial_decomposition)
        elif self.init.lower() == 'precomputed':
            self._check_valid_components(initial_decomposition)
            self.decomposition = initial_decomposition
        elif Path(self.init).is_file():
            self.load_checkpoint(self.init)
        else:
            # TODO: better message
            raise ValueError('Init method must be either `random`, `cp`, `svd`, `from_checkpoint`, `precomputed` or a path to a checkpoint.')

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

        Arguments:
        ----------
        X : np.ndarray
            The tensor or list of tensor slices to fit a PARAFAC2 model to. 
            The list indices (or first mode) correspond to the C-mode in
            the following equation

            .. math::

                X_k = A diag(c_k) B_k^T
            
            This will be changed in a later version so that the first mode
            is the evolving mode.
        y : None
            Ignored, included to follow sklearn standards.
        max_its : int (optional)
            If set, then this will override the class's max_its.
        initial_decomposition : tenkit.decomposition.decompositions.Parafac2Tensor or str
            The initial KruskalTensor (init=precomputed) to use or the path of the 
            logfile to load (init=from_file).
        """

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
        return utils.slice_SSE(self.X, self.reconstructed_X)

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

    # TODO: Change name of this function
    def _update_projection_matrices(self):
        K = self.X_shape[2]

        for k in range(K):
            A = self.decomposition.A
            C = self.decomposition.C
            blueprint_B = self.decomposition.blueprint_B

            self.decomposition.projection_matrices[k][...] = base.orthogonal_solve(
                (C[k]*A)@blueprint_B.T,
                self.X[k]
            ).T

            # Should_keep = diag([1, 1, ..., 1, 0, 0, ..., 0]) -> the zeros correspond to small singular values
            # Following Rasmus Bro's PARAFAC2 MATLAB script, which sets P_k = Q_k(Q_k'Q_k)^(-0.5) (line 524)
            #      Where the power is done by truncating very small singular values (for numerical stability)


class Parafac2_ALS(BaseParafac2):
    r"""Decomposer for Parafac2 with ALS optimization

    Arguments:
    ----------
    rank: int
        Number of components 
    max_its: int (optional, default=1000)
        Maximum number of iterations for fitting the model. 
        Can be overwritten by the ``fit`` method.
    convergence_tol: float (optional, default=1e-6)
        Minimum relative function change between two consequetive
        iterations for the model to continue fitting. Computed as
        
        .. math::

            \frac{L(X_i) - L(X_{i+1}) }{L(X_{i+1})},
        
        where :math:`L` is the loss (Sum squared error) and :math:`X_i`
        is the decomposition at iteration :math:`i`.
    init: str (optional, default = 'random')
        Initialisation scheme for the decomposer (case insensitive). Options:
    
          * Random: Initiate the decomposition as a random Parafac2 tensor
          * SVD: Use the SVD to find the factor matrices
          * CP: Run 20 CP iterations and use that decomposition as initial
          Parafac2 tensor (QR on the evolving mode to split into projection
          and blueprint matrices)
          * from_checkpoint: Load the last iteration in the specified
          checkpoint file (alternatively, just the path to the decomposition file)
          * Precomputed: Use already existing Parafac2 tensor to initialise
          the decomposition. 
    logger: list(Logger) (optional, default=None)
        List of loggers, each logger should implement a ``log`` method
        that takes a decomposer as input and a ``write_to_hdf5_group``
        method that stores the log in a hdf5 group. See 
        ``tenkit.logging.BaseLogger`` for interface.
    checkpoint_frequency: int (optional, default=None)
        How often the decomposer should store the decomposition and
        logs to disk. If None or negative, will only
        checkpoint the last iteration. 
    checkpoint_path: str or Path (optional, default=None)
        Where to store the log HDF5 file. If None, then the checkpoints
        and logs are not stored to disk.
    print_frequency: int (optional, default=None)
        How often convergence information should be printed in the terminal.
        None and negative values leads to no printing.
    cp_updates_per_it: int (optional, default=5)
        Number of CP iterations to run between each projection matrix update.
    non_negativity_constraints: list(bool) (optional, default=None)
        If nth element in the list is True, the nth mode is constrained to be
        non-negative. If None, no modes are constrained.
        Note: The evolving mode cannot be constrained. 
    ridge_penalties: list(float) (optional, default=None)
        Regularisation parameters. The loss is regularised such that

        .. math::

            L_{reg} = L + \sum_i \lambda_i ||U_i||_F^2

        Where :math:`L_{reg}` is the regularised loss, :math:`L` is the unregularised
        loss, :math:`\lambda_i` is the ith element in ``ridge_penalites`` and :math:`U_i`
        is the ith factor matrix (or blueprint factor matrix for the evolving mode). 
        If None, no modes are regularised.
    orthonormality_constraints: list(bool) (optional, default=None)
        If nth element in the list is True, the nth mode is constrained to be
        orthonormal. If None, no modes are constrained. Note: all modes should not be
        orhtonormal and the evolving mode cannot be constrained. 
    """
    def __init__(
        self,
        rank,
        max_its=1000, 
        convergence_tol=1e-6, 
        init='random',
        loggers=None,
        checkpoint_frequency=None,
        checkpoint_path=None,
        print_frequency=10,
        cp_updates_per_it=5,
        non_negativity_constraints=None,
        ridge_penalties=None,
        orthonormality_constraints=None,
    ):
        super().__init__(
            rank,
            max_its=max_its,
            convergence_tol=convergence_tol,
            init=init,
            loggers=loggers,
            checkpoint_frequency=checkpoint_frequency,
            checkpoint_path=checkpoint_path,
            print_frequency=print_frequency,
        )
        self.non_negativity_constraints = non_negativity_constraints
        if self.non_negativity_constraints is None:
            self.non_negativity_constraints = [False, False, False]
        if self.non_negativity_constraints[1]:
            warnings.warn(
                "Non negativity constraints on the evolving (second) mode will only"
                " be imposed on the blueprint matrix, not the evolving components.",
                RuntimeWarning
            )

        self.cp_updates_per_it = cp_updates_per_it
        self.ridge_penalties = ridge_penalties
        self.orthonormality_constraints = orthonormality_constraints

    def init_random(self):
        """Random initialisation of the factor matrices
        """
        self.decomposition = self.DecompositionType.random_init(self.X_shape, rank=self.rank, non_negativity=self.non_negativity_constraints)

    def _init_fit(self, X, max_its, initial_decomposition):
        super()._init_fit(X=X, max_its=max_its, initial_decomposition=initial_decomposition)
        self.prev_loss = self.loss
        self._rel_function_change = np.inf

    def _prepare_cp_decomposer(self):
        self.cp_decomposition = decompositions.KruskalTensor([self.decomposition.A, self.decomposition.blueprint_B, self.decomposition.C])
        self.cp_decomposer = cp.CP_ALS(
            self.rank,
            max_its=np.inf, 
            convergence_tol=0,
            print_frequency=-1, 
            non_negativity_constraints=self.non_negativity_constraints,
            ridge_penalties=self.ridge_penalties,
            orthonormality_constraints=self.orthonormality_constraints,
            init='precomputed'
        )
        self.cp_decomposer._init_fit(X=self.projected_X, max_its=np.inf, initial_decomposition=self.cp_decomposition)

    def _fit(self):

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
        loss = self.loss
        self._rel_function_change = (self.prev_loss - loss)/self.prev_loss
        self.prev_loss = loss

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

    @property
    def loss(self):
        loss = self.SSE
        if self.ridge_penalties is not None:
            factor_matrices = [self.decomposition.A, self.decomposition.blueprint_B, self.decomposition.C]
            for factor, penalty in zip(factor_matrices, self.ridge_penalties):
                loss += penalty*np.linalg.norm(factor)**2
        return loss
