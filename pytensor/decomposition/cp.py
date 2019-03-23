from abc import abstractmethod

import h5py
import numpy as np
from scipy.optimize import nnls

from .base_decomposer import BaseDecomposer
from .. import base
from ..utils import normalize_factors


class BaseCP(BaseDecomposer):
    DecompositionType = base.KruskalTensor
    def __init__(
        self,
        rank,
        max_its,
        convergence_tol=1e-10,
        init='random',
        loggers=None,
        checkpoint_period=None,
        checkpoint_name=None
    ):
        if loggers is None:
            loggers = []
        if checkpoint_period is None:
            checkpoint_period = -1

        self.rank = rank
        self.max_its = max_its
        self.convergence_tol = convergence_tol
        self.init = init
        self.loggers = loggers
        self.checkpoint_period = checkpoint_period
        self.checkpoint_name = checkpoint_name

    def init_random(self):
        """Random initialisation of the factor matrices.

        Each element of the factor matrices are taken from a standard normal distribution.
        """
        self.decomposition = self.DecompositionType.random_init(self.X.shape, rank=self.rank)
    
    def init_svd(self):
        """SVD initialisation of the factor matrices.
        """
        n_modes = len(self.X.shape)
        factor_matrices = []
        if self.rank > min(self.X.shape):
            raise ValueError(
                "SVD initialisation does not work when rank is larger than the smallest dimension of X."
                f" (rank:{self.rank}, dimensions: {self.X.shape})"
            )
        for i in range(n_modes):
            u, _, _ = np.linalg.svd(base.unfold(self.X, i))

            factor_matrices.append(u[:, :self.rank])
        
        self.decomposition = base.KruskalTensor(factor_matrices)
 
    def _check_valid_components(self, decomposition):
        """Check if provided factor matrices have correct shape.
        """
        for i, factor_matrix in enumerate(decomposition.factor_matrices):
            len_, rank = factor_matrix.shape
            if rank != self.rank:
                raise ValueError(
                    f"The specified factor matrix' rank ({rank}) does not agree with the model's rank ({self.rank})."
                )

            xlen = self.X.shape[i]
            if len_ != xlen:
                raise ValueError(
                    f"The length of component {i} ({len_}) is not the same as the length of X's dimension {i} ({xlen})."
                )

    def init_components(self, initial_decomposition=None):
        """
        """
        if (initial_decomposition is not None and 
            self.init.lower() not in  ['precomputed', 'from_checkpoint']):
            raise Warning(f'Precomputed components were passed even though {self.init} initialisation is used.'
                           'The precomputed components will therefore be disregarded')

        if self.init.lower() == 'random':
            self.init_random()

        elif self.init.lower() == 'svd':
            self.init_svd()

        elif self.init.lower() == 'from_checkpoint':
            with h5py.File(initial_decomposition) as h5:
                self.current_iteration = h5.attrs['final_iteration']
                checkpoint_group = h5[f'checkpoint_{self.current_iteration:05d}']

                initial_decomposition = self.DecompositionType.load_from_hdf5_group(checkpoint_group)

            self._check_valid_components(initial_decomposition)
            self.decomposition = initial_decomposition

        elif self.init.lower() == 'precomputed':
            self._check_valid_components(initial_decomposition)
            self.decomposition = initial_decomposition

        else:
            raise ValueError('Init method must be either `random`, `svd`, `from_checkpoint` or `precomputed`.')

    @abstractmethod
    def _fit(self):
        pass

    def _init_fit(self, X, max_its, initial_decomposition):
        self.set_target(X)
        self.current_iteration = 0
        self.init_components(initial_decomposition=initial_decomposition)
        if max_its is not None:
            self.max_its = max_its

    def fit(self, X, y=None, *, max_its=None, initial_decomposition=None):
        """Fit a CP model. Precomputed components must be specified if init method is `precomputed`.

        Arguments:
        ----------
        X : np.ndarray
            The tensor to fit
        y : None
            Ignored, included to follow sklearn standards.
        max_its : int (optional)
            If set, then this will override the class's max_its.
        initial_decomposition : pytensor.base.KruskalTensor or str
            The initial KruskalTensor (init=precomputed) to use or the path of the 
            logfile to load (init=from_file).
        """
        self._init_fit(
            X=X, max_its=max_its, initial_decomposition=initial_decomposition
        )
        self._fit()

    def fit_transform(self, X, y=None, *, max_its=None, initial_decomposition=None):
        """Fit a CP model and return kruskal tensor. 
        
        Precomputed components must be specified if init method is `precomputed`.

        Arguments:
        ----------
        X : np.ndarray
            The tensor to fit
        y : None
            Ignored, included to follow sklearn standards.
        max_its : int (optional)
            If set, then this will override the class's max_its.
        initial_decomposition : tuple
            A tuple parametrising a Kruskal tensor.
            The first element is a list of factor matrices and the second element is an array containing the weights.
        """
        self.fit(X=X, y=y, max_its=max_its, initial_decomposition=initial_decomposition)
        return self.decomposition

    def loss(self):
        return self.SSE #TODO: skal det være property?
   
    def _fit(self):
        return 1 - self.SSE/(self.X_norm**2)
    
    @property
    def reconstructed_X(self):
        # Todo: Cache this
        return self.decomposition.construct_tensor()

    @property
    def factor_matrices(self):
        return self.decomposition.factor_matrices
    
    @property
    def weights(self):
        return self.decomposition.weights
    
    def store_checkpoint(self):
        with h5py.File(self.checkpoint_name, 'a') as h5:
            h5.attrs['final_iteration'] = self.current_iteration
            checkpoint_group = h5.create_group(f'checkpoint_{self.current_iteration:05d}')
            self.decomposition.store_in_hdf5_group(checkpoint_group)
            for logger in self.loggers:
                logger.write_to_hdf5_group(h5)


class CP_ALS(BaseCP):
    """CP (CANDECOMP/PARAFAC) decomposition using Alternating Least Squares."""

    def __init__(
        self,
        rank,
        max_its,
        convergence_tol=1e-10,
        init='random',
        loggers=None,
        checkpoint_period=None,
        checkpoint_name=None,
        print_frequency=1,
        non_negativity_constraints=None
    ):
        super().__init__(
            rank=rank,
            max_its=max_its,
            convergence_tol=convergence_tol,
            init=init,
            loggers=loggers,
            checkpoint_period=checkpoint_period,
            checkpoint_name=checkpoint_name
        )
        self.print_frequency = print_frequency
        self.non_negativity_constraints = non_negativity_constraints

    def _init_fit(self, X, max_its, initial_decomposition):
        super()._init_fit(X=X, max_its=max_its, initial_decomposition=initial_decomposition)
        self._rel_function_change = np.inf
        self.prev_SSE = self.SSE

    def _compute_V(self, skip_mode):
        """Compute left hand side of least squares problem."""

        V = np.ones((self.rank, self.rank))
        for i, factor in enumerate(self.factor_matrices):
            if i == skip_mode:
                continue
            V *= factor.T @ factor
        return V

    def _update_als_factor(self, mode):
        """Solve least squares problem to get factor for one mode."""
        
        self.decomposition.normalize_components()
        V = self._compute_V(mode)
        n = np.prod(V.shape)

        _rhs = base.matrix_khatri_rao_product(self.X, self.factor_matrices, mode).T
        U, S, W = np.linalg.svd(V.T, full_matrices=False)
        new_factor = (W.T @ np.diag(1/(S + 1e-5/n)) @ U.T @ _rhs).T  #TODO er det denne måten vi vil gjøre det?

        self.factor_matrices[mode][...] = new_factor

    def _update_als_factor_non_negative(self, mode):
        """Solve non negative least squares problem to get factor for one mode."""

        self.decomposition.normalize_components()
        V = self._compute_V(mode)
        _rhs = base.matrix_khatri_rao_product(self.X, self.factor_matrices, mode).T

        new_factor_tmp = np.zeros_like(self.factor_matrices[mode].T)
        for j in range(_rhs.shape[1]):
            new_factor_tmp[:,j],_ = nnls(V.T, _rhs[:,j]) 

        new_factor = new_factor_tmp.T

        self.factor_matrices[mode] = new_factor

    def _update_als_factors(self):
        """Updates factors with alternating least squares."""
        num_modes = len(self.X.shape) # TODO: Should this be cashed?
        for mode in range(num_modes):
            if self.non_negativity_constraints[mode]:
                self._update_als_factor_non_negative(mode) 
            else:
                self._update_als_factor(mode)
    
    def _update_convergence(self):
        self._rel_function_change = (self.prev_SSE - self.SSE)/self.prev_SSE
        self.prev_SSE = self.SSE 

    def _fit(self):
        """Fit a CP model with Alternating Least Squares.
        """
        # TODO: logger?

        if self.non_negativity_constraints is None:
            self.non_negativity_constraints = [False]*len(self.factor_matrices)

        for it in range(self.max_its):
            if abs(self._rel_function_change) < self.convergence_tol:
                break
            self._update_als_factors()
            self._update_convergence()
            for logger in self.loggers:
                logger.log(self)
            
            if ((it+1) % self.checkpoint_period == 0) and (self.checkpoint_period > 0):
                self.store_checkpoint()

            if it % self.print_frequency == 0 and self.print_frequency > 0:
                print(f'{it}: The MSE is {self.MSE:4f}, f is {self.loss():4f}, improvement is {self._rel_function_change:4g}')

            self.current_iteration += 1


