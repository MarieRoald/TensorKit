from abc import abstractmethod
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import nnls

from .base_decomposer import BaseDecomposer
from . import decompositions
from .. import base
from ..utils import normalize_factors


class BaseCP(BaseDecomposer):
    DecompositionType = decompositions.KruskalTensor
    def __init__(
        self,
        rank,
        max_its,
        convergence_tol=1e-10,
        init='random',
        loggers=None,
        checkpoint_frequency=None,
        checkpoint_path=None,
        ridge_penalties=None,
    ):
        super().__init__(
            loggers=loggers,
            checkpoint_frequency=checkpoint_frequency,
            checkpoint_path=checkpoint_path
        )
        self.rank = rank
        self.max_its = max_its
        self.convergence_tol = convergence_tol
        self.init = init
        self.ridge_penalties = ridge_penalties

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
        
        self.decomposition = self.DecompositionType(factor_matrices)
 
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
        """Initialize the components with the initialization method in `self.init`. 
        If `self.init` is not 'random' or 'svd' initial_decomposition must be provided.

        Arguments:
        ----------
        initial_decompostion: tenkit.decomposition.decompositions.KruskalTensor or str (optional)
            The initial KruskalTensor (init=precomputed) to use or the path of the 
            logfile to load (init=from_file).
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
            self.load_checkpoint(initial_decomposition)

        elif self.init.lower() == 'precomputed':
            self._check_valid_components(initial_decomposition)
            self.decomposition = initial_decomposition

        elif Path(self.init).is_file():
            self.load_checkpoint(self.init)

        else:
            raise ValueError('Init method must be either `random`, `svd`, `from_checkpoint` or `precomputed`.')

    @abstractmethod
    def _fit(self):
        pass

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
        initial_decomposition : tenkit.decomposition.decompositions.KruskalTensor or str
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

    @property
    def loss(self):
        loss = self.SSE #TODO: skal det være property?
        if self.ridge_penalties is not None:
            for ridge, factor_matrix in zip(self.ridge_penalties, self.factor_matrices):
                print('loss')
                print(ridge)
                print(np.prod(factor_matrix.shape))
                ridge /= np.prod(factor_matrix.shape)
                loss += ridge*np.linalg.norm(factor_matrix)**2
        return loss
   
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


class CP_ALS(BaseCP):
    """CP (CANDECOMP/PARAFAC) decomposition using Alternating Least Squares."""

    def __init__(
        self,
        rank,
        max_its,
        convergence_tol=1e-10,
        init='random',
        loggers=None,
        checkpoint_frequency=None,
        checkpoint_path=None,
        print_frequency=10,
        non_negativity_constraints=None,
        ridge_penalties=None
    ):
        super().__init__(
            rank=rank,
            max_its=max_its,
            convergence_tol=convergence_tol,
            init=init,
            loggers=loggers,
            checkpoint_frequency=checkpoint_frequency,
            checkpoint_path=checkpoint_path,
            ridge_penalties=ridge_penalties
        )
        self.print_frequency = print_frequency
        self.non_negativity_constraints = non_negativity_constraints

    def _init_fit(self, X, max_its, initial_decomposition):
        super()._init_fit(X=X, max_its=max_its, initial_decomposition=initial_decomposition)
        self.decomposition.reset_weights()
        if self.non_negativity_constraints is None:
            self.non_negativity_constraints = [False]*len(self.factor_matrices)

        self._rel_function_change = np.inf
        self.prev_SSE = self.SSE

    def _get_als_lhs(self, skip_mode):
        """Compute left hand side of least squares problem."""
        V = np.ones((self.rank, self.rank))
        for i, factor in enumerate(self.factor_matrices):
            if i == skip_mode:
                continue
            V *= factor.T @ factor
        return V
    
    def _get_als_rhs(self, mode):
        return base.matrix_khatri_rao_product(self.X, self.factor_matrices, mode)

    def _get_rightsolve(self, mode):
        rightsolve = base.rightsolve
        if self.non_negativity_constraints[mode]:
            rightsolve = base.non_negative_rightsolve
        
        if self.ridge_penalties is not None:
            ridge_penalty = self.ridge_penalties[mode]
            fm_shape = np.prod(self.factor_matrices[mode].shape)
            print('get rightsolve', mode)
            print(ridge_penalty)
            print(fm_shape)
            rightsolve = base.add_rightsolve_ridge(rightsolve, ridge_penalty/fm_shape)

        return rightsolve

    def _update_als_factor(self, mode):
        """Solve least squares problem to get factor for one mode."""
        lhs = self._get_als_lhs(mode)
        rhs = self._get_als_rhs(mode)

        rightsolve = self._get_rightsolve(mode)

        new_factor = rightsolve(lhs, rhs)
        self.factor_matrices[mode][...] = new_factor

    def _update_als_factors(self):
        """Updates factors with alternating least squares."""
        num_modes = len(self.X.shape) # TODO: Should this be cashed?
        for mode in range(num_modes):
            self._update_als_factor(mode)
   
    def _update_convergence(self):
        self._rel_function_change = (self.prev_SSE - self.SSE)/self.prev_SSE
        self.prev_SSE = self.SSE 

    def _fit(self):
        """Fit a CP model with Alternating Least Squares.
        """
        for it in range(self.max_its - self.current_iteration):
            if abs(self._rel_function_change) < self.convergence_tol:
                break
            self._update_als_factors()
            self._update_convergence()

            if self.current_iteration % self.print_frequency == 0 and self.print_frequency > 0:
                print(f'    {self.current_iteration}: The MSE is {self.MSE:4g}, f is {self.loss:4g}, improvement is {self._rel_function_change:4g}')

            self._after_fit_iteration()


        if (it+1) % self.checkpoint_frequency != 0:
            self.store_checkpoint()

    def init_random(self):
        super().init_random()
        if self.non_negativity_constraints is None:
            return 
        
        for mode, non_negativity in enumerate(self.non_negativity_constraints):
            fm = self.decomposition.factor_matrices[mode]
            if non_negativity:
                fm[...] = np.abs(fm)
