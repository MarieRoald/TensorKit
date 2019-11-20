import numpy as np

import base
import utils


class BaseTensorDecomposition:
    def __init__(self):
        pass

    def decompose(self, tensor):
        pass

    def compose_from_factors(self, factors):
        pass

    def compute_loss(self, factors):
        pass


class Base_CP(BaseTensorDecomposition):
    def __init__(self, rank, init_scheme):
        pass

    def init_factors(self, tensor, rank, init_scheme="random"):
        """Initialize factor matrices.

        Method of initialization is decided by `init_scheme`. If `init_scheme == 'random'`, the 
        factor matrices are initialized randomly. If `init_scheme == 'svd'`, the nth factor
        matrix is initialized as the `rank` left singular vectors of the input tensor unfolded
        in the nth mode. 
        
        Parameters:
        -----------
        tensor: np.ndarray
            Tensor we are attempting to model with CP.
        rank: int
            The number of compononents in the model
        init_scheme: str (optional)
            String defining which init scheme to use. Must be 'random' (default) or 'svd'

        """
        if init_scheme == "random":
            factors = self._random_init(tensor, rank)
        elif init_scheme == "svd":
            factors = self._svd_init(tensor, rank)
        else:
            raise ValueError(
                f'Unknown init_scheme "{init_scheme}", use "svd" or "random"'
            )

        weights = np.ones((len(tensor.shape), rank))
        return [utils.normalize_factor(f)[0] for f in factors], weights

    def _random_init(self, tensor, rank):
        """Random initialization of factor matrices.

        Each element in the factor matrices is sampled from the
        standard normal distribution with np.random.random.randn
        
        Parameters:
        -----------
        tensor: np.ndarray
            Tensor we are attempting to model with CP.
        rank: int
            The number of compononents for the model

        Returns:
        --------
        list of np.ndarrays:
            Initialized factor matrices. Each factor matrix F_i has
            shape (s, rank) where s is the length of the ith mode of
            the input tensor
        """
        factors = [np.random.randn(s, rank) for s in tensor.shape]
        return factors

    def _svd_init(self, tensor, rank):
        """SVD based initialization of factor matrices.

        Initializes each factor F_i as the `rank` first singular vectors
        of the tensor unfolded along the corresponding mode. 
        Uses np.linalg.svd to calculate the singular value decomposition.

        Parameters:
        -----------
        tensor: np.ndarray
            Tensor we are attempting to model with CP.
        rank: int
            The number of compononents for the model

        Returns:
        --------
        list of np.ndarrays:
            Initialized factor matrices. Each factor matrix F_i has
            shape (s, rank) where s is the length of the ith mode of
            the input tensor
        """
        n_modes = len(tensor.shape)
        factors = []
        for i in range(n_modes):
            u, s, vh = np.linalg.svd(base.unfold(tensor, i))
            factors.append(u[:, :rank])
        return factors

    def compute_loss(self, tensor, rank):
        pass

    def compose_from_factors(self, factors):
        pass


class CP_als(Base_CP):
    def __init__(self, rank, init_scheme, max_it, tol):
        pass

    def als_cycle(self, factors, tensor):
        pass

    def als_it(self, factors, tensor):
        pass

    def decompose(self, tensor):
        pass
