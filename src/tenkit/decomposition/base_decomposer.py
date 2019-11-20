"""
Contains the base class for all decomposition methods in TensorKit
"""


from abc import ABC, abstractmethod, abstractproperty

import h5py
import numpy as np

from . import decompositions

__author__ = "Marie Roald & Yngve Mardal Moe"


class BaseDecomposer(ABC):
    r"""Base class for all TensorKit decomposer objects

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
    """
    DecompositionType = decompositions.BaseDecomposedTensor
    @abstractmethod
    def __init__(
        self,
        max_its=1000,
        convergence_tol=1e-6, 
        loggers=None,
        checkpoint_frequency=None,
        checkpoint_path=None,
        print_frequency=None,
    ):
        self.max_its = max_its
        self.convergence_tol = convergence_tol
        if checkpoint_frequency is None:
            checkpoint_frequency = -1
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_path = checkpoint_path

        if print_frequency is None:
            print_frequency = -1
        self.print_frequency = print_frequency 

        if loggers is None:
            loggers = []
        self.loggers = loggers

    @abstractmethod
    def init_components(self, initial_decomposition=None):
        pass

    @abstractproperty
    def reconstructed_X(self):
        pass
    
    def set_target(self, X):
        """Set target for fitting of model.

        Arguments
        ---------
        X : np.ndarray
            The tensor to fit the model to
        """
        self.X = X
        self.X_norm = np.linalg.norm(X)

    @property
    def explained_variance(self):
        # TODO: Cache result
        return 1 - self.SSE/self.X_norm**2

    @property
    def SSE(self):
        """Sum Squared Error"""
        # TODO: Cache result
        return np.linalg.norm(self.X - self.reconstructed_X)**2
    
    @property
    def MSE(self):
        """Mean Squared Error"""
        # TODO: Cache result
        return self.SSE/np.prod(self.X.shape)
    
    @property
    def RMSE(self):
        """Root Mean Squared Error"""
        # TODO: Cache result
        return np.sqrt(self.MSE)

    # TODO: Property?
    # TODO: Hvis det blir property, fiks logger!
    @abstractproperty
    def loss(self):
        pass
   
    @abstractmethod
    def _fit(self):
        pass

    def _init_fit(self, X, max_its=None, initial_decomposition=None):
        """Prepare model for fitting by initialising and setting target and max_its.

        Arguments
        ---------
        X : np.ndarray
            The tensor to fit the model to
        max_its : int (optional)
            If set, then this will override the class's max_its.
        initial_decomposition : BaseDecomposedTensor (optional)
            None (default) or a BaseDemposedTensor object containig the 
            initial decomposition. If class's init is not 'precomputed' it is ignored.
        """
        self.current_iteration = 0
        self.set_target(X)
        self.init_components(initial_decomposition=initial_decomposition)
        if max_its is not None:
            self.max_its = max_its

    def fit(self, X, y=None, *, max_its=None, initial_decomposition=None):
        """Fit a tensor decomposition model. 
        
        Precomputed components must be specified if init method is 'precomputed'.

        Arguments
        ---------
        X : np.ndarray
            The tensor to fit the model to
        y : None
            Ignored, included to follow sklearn standards.
        max_its : int (optional)
            If set, then this will override the class's max_its.
        initial_decomposition : BaseDecomposedTensor (optional)
            None (default) or a BaseDemposedTensor object containig the 
            initial decomposition. If class's init is not 'precomputed' it is ignored.
        """
        self._init_fit(X=X, max_its=max_its, initial_decomposition=initial_decomposition)
        self._fit()
    
    def continue_fit(self, max_its=None):
        """Continue training an allready fitted model.

        Arguments
        ---------
        max_its : int (optional)
            If set, then this will override the class's max_its. s
        """
        if max_its is not None:
            self.max_its = max_its
        self._fit()
    
    def store_checkpoint(self):
        with h5py.File(self.checkpoint_path, 'a') as h5:
            if 'checkpoint_its' not in h5.attrs:
                h5.attrs['checkpoint_its'] = [self.current_iteration]
            else:
                h5.attrs['checkpoint_its'] = [*h5.attrs['checkpoint_its'], self.current_iteration]

            h5.attrs['final_iteration'] = self.current_iteration
            h5.attrs['decomposition_type'] = type(self).__name__
            checkpoint_group = h5.create_group(f'checkpoint_{self.current_iteration:05d}')
            self.decomposition.store_in_hdf5_group(checkpoint_group)

            for logger in self.loggers:
                logger.write_to_hdf5_group(h5)
    
    @abstractmethod
    def _check_valid_components(self, decomposition):
        pass

    def load_checkpoint(self, checkpoint_path, load_it=None):
        """Load the specified checkpoint at the given iteration.

        If ``load_it=None``, then the latest checkpoint will be used.
        """
        # TODO: classmethod, dump all params. Requires major refactoring.
        with h5py.File(checkpoint_path) as h5:
            if 'final_iteration' not in h5.attrs:
                raise ValueError(f'There is no checkpoints in {checkpoint_path}')

            if load_it is None:
                load_it = h5.attrs['final_iteration']
            self.current_iteration = load_it

            group_name = f'checkpoint_{load_it:05d}'
            if group_name not in h5:
                raise ValueError(f'There is no checkpoint {group_name} in {checkpoint_path}')

            checkpoint_group = h5[f'checkpoint_{load_it:05d}']
            initial_decomposition = self.DecompositionType.load_from_hdf5_group(checkpoint_group)

        self._check_valid_components(initial_decomposition)
        self.decomposition = initial_decomposition
    
    def _after_fit_iteration(self):
        for logger in self.loggers:
            logger.log(self)

        it = self.current_iteration
        if ((it+1) % self.checkpoint_frequency == 0) and (self.checkpoint_frequency > 0):
            self.store_checkpoint()

        self.current_iteration += 1
