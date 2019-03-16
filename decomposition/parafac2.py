from abc import abstractmethod
import numpy as np
from .base_decomposer import BaseDecomposer
from ..utils import normalize_factors
from .. import base


class BaseParafac2(BaseDecomposer):
    def __init__(self, rank, max_its, convergence_tol=1e-10, init='random',  evolve_mode=0, evolve_over=1):
        self.rank = rank
        self.max_its = max_its
        self.convergence_tol = convergence_tol
        self.init = init
        self.evolve_mode = evolve_mode
        self.evolve_over = evolve_over

    def init_random(self):
        """Random initialisation of the factor matrices
        """
        self.decomposition = base.Parafac2Tensor.random_init(self.X_shape, rank=self.rank)
    