from abc import abstractmethod
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import nnls

from .base_decomposer import BaseDecomposer
from . import decompositions
from .. import base
from ..utils import normalize_factors


__all__ = ['CP_ALS']


class BaseCP(BaseDecomposer):
    r"""CP (CANDECOMP/PARAFAC) decomposition using Alternating Least Squares.

    Arguments:
    ----------
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
    ridge_penalties: list(float) (optional, default=None)
        Regularisation parameters. The loss is regularised such that

        .. math::

            L_{reg} = L + \sum_i \lambda_i ||U_i||_F^2

        Where :math:`L_{reg}` is the regularised loss, :math:`L` is the unregularised
        loss, :math:`\lambda_i` is the ith element in ``ridge_penalites`` and :math:`U_i`
        is the ith factor matrix (or blueprint factor matrix for the evolving mode). 
        If None, no modes are regularised.
    """
    DecompositionType = decompositions.KruskalTensor
    def __init__(
        self,
        rank,
        max_its=1000,
        convergence_tol=1e-6,
        rel_loss_tol=1e-10,
        init='random',
        loggers=None,
        checkpoint_frequency=None,
        checkpoint_path=None,
        ridge_penalties=None,
        print_frequency=None,
    ):
        super().__init__(
            loggers=loggers,
            checkpoint_frequency=checkpoint_frequency,
            checkpoint_path=checkpoint_path,
            print_frequency=print_frequency,
            max_its=max_its,
            convergence_tol=convergence_tol,
        )
        self.rank = rank
        self.init = init
        self.ridge_penalties = ridge_penalties
        self.rel_loss_tol = rel_loss_tol

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

    @property
    def SSE(self):
        """Sum Squared Error"""
        if hasattr(self, '_last_updated_mode') and self._last_updated_mode is not None:
            # ||X - Y||_F^2 = ||X||_F^2 + ||Y||_F^2 - 2<X, Y>_F
            # Y = [U_0, U_1, U_2], <X, Y> = sum(U_i*mttkrp(X, Y, skip=i))
            
            return (
                self.X_norm**2
                 + np.linalg.norm(self.reconstructed_X)**2
                 - 2*self._inner_prod_X_reconstructed_X
            )
            
        return np.linalg.norm(self.X - self.reconstructed_X)**2

    @property
    def _inner_prod_X_reconstructed_X(self):
        M = self.factor_matrices[self._last_updated_mode]*self._matrix_khatri_rao_product_cache
        return np.sum(self.weights*M.sum(0), axis=0)

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
        loss = self.SSE
        if self.ridge_penalties is not None:
            for ridge, factor_matrix in zip(self.ridge_penalties, self.factor_matrices):
                loss += ridge*np.linalg.norm(factor_matrix)**2
        return loss
   
    def _fit(self):
        return 1 - self.SSE/(self.X_norm**2)
    
    @property
    def reconstructed_X(self):
        return self.decomposition.construct_tensor()

    @property
    def factor_matrices(self):
        return self.decomposition.factor_matrices

    @property
    def weights(self):
        return self.decomposition.weights

class CP_ALS(BaseCP):
    r"""CP (CANDECOMP/PARAFAC) decomposition using Alternating Least Squares.

    Arguments:
    ----------
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
    non_negativity_constraints: list(bool) (optional, default=None)
        If nth element in the list is True, the nth mode is constrained to be
        non-negative. If None, no modes are constrained.
        The evolving mode cannot be constrained. 
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
        orhtonormal.
    """

    def __init__(
        self,
        rank,
        max_its=1000,
        convergence_tol=1e-6,
        rel_loss_tol=1e-10,
        init='random',
        loggers=None,
        checkpoint_frequency=None,
        checkpoint_path=None,
        print_frequency=None,
        non_negativity_constraints=None,
        ridge_penalties=None,
        orthonormality_constraints=None,
    ):
        super().__init__(
            rank=rank,
            max_its=max_its,
            convergence_tol=convergence_tol,
            rel_loss_tol=rel_loss_tol,
            init=init,
            loggers=loggers,
            checkpoint_frequency=checkpoint_frequency,
            checkpoint_path=checkpoint_path,
            ridge_penalties=ridge_penalties,
            print_frequency=print_frequency
        )
        self.non_negativity_constraints = non_negativity_constraints
        self.orthonormality_constraints = orthonormality_constraints


    def _init_fit(self, X, max_its, initial_decomposition):
        super()._init_fit(X=X, max_its=max_its, initial_decomposition=initial_decomposition)
        self.decomposition.reset_weights()
        if self.non_negativity_constraints is None:
            self.non_negativity_constraints = [False]*len(self.factor_matrices)
        if self.orthonormality_constraints is None:
            self.orthonormality_constraints = [False]*len(self.factor_matrices)
        for mode, (orthogonality) in enumerate(self.orthonormality_constraints):
            fm = self.decomposition.factor_matrices[mode]
            if orthogonality:
                self.decomposition.factor_matrices[mode] = np.linalg.qr(fm)[0]

        self._rel_function_change = np.inf
        self.prev_SSE = self.SSE

        self._matrix_khatri_rao_product_cache = None
        # self._matrix_khatri_rao_product_cache = [np.empty_like(factor) for factor in self.factor_matrices]

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
        
        if self.orthonormality_constraints[mode]:
            if self.non_negativity_constraints[mode]:
                raise ValueError('Cannot perform nonnegative orthogonal solve')

            rightsolve = base.orthogonal_rightsolve

        if self.ridge_penalties is not None:
            ridge_penalty = self.ridge_penalties[mode]
            # fm_shape = np.prod(self.factor_matrices[mode].shape)
            rightsolve = base.add_rightsolve_ridge(rightsolve, ridge_penalty)#/fm_shape)
        

        return rightsolve

    def _update_als_factor(self, mode):
        """Solve least squares problem to get factor for one mode."""
        lhs = self._get_als_lhs(mode)
        rhs = self._get_als_rhs(mode)

        rightsolve = self._get_rightsolve(mode)

        self._last_updated_mode = mode
        self._matrix_khatri_rao_product_cache = rhs

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
            rel_loss = self.SSE/self.X_norm**2
            if (
                abs(self._rel_function_change) < self.convergence_tol 
                or rel_loss < self.rel_loss_tol
            ):
                break
            self._update_als_factors()
            self._update_convergence()

            if self.print_frequency > 0 and self.current_iteration % self.print_frequency == 0:
                print(f'    {self.current_iteration}: The MSE is {self.MSE:4g}, f is {self.loss:4g}, improvement is {self._rel_function_change:4g}')

            self._after_fit_iteration()

        if ((it+1) % self.checkpoint_frequency != 0) and (self.checkpoint_frequency > 0):
            self.store_checkpoint()

    def init_random(self):
        super().init_random()
        if self.non_negativity_constraints is None:
            return 
        
        for mode, non_negativity in enumerate(self.non_negativity_constraints):
            fm = self.decomposition.factor_matrices[mode]
            if non_negativity:
                fm[...] = np.abs(fm)
