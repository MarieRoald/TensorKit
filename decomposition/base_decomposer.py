"""
Contains the base class for all decomposition methods in PyTensor
"""


from abc import ABC, abstractmethod, abstractproperty
import numpy as np


__author__ = "Marie Roald & Yngve Mardal Moe"


class BaseDecomposer(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def fit(self, X, y=None, *, max_its):
        pass
    
    @abstractproperty
    def reconstructed_X(self):
        pass
    
    def set_target(self, X):
        self.X = X

    @property
    def explained_variance(self):
        # TODO: Cache result
        return 1 - self.SSE/np.linalg.norm(self.X)**2

    @property
    def SSE(self):
        # TODO: Cache result
        return np.linalg.norm(self.X - self.reconstructed_X)**2
    
    @property
    def MSE(self):
        # TODO: Cache result
        return self.SSE/np.prod(self.X.shape)
    
    @property
    def RMSE(self):
        # TODO: Cache result
        return np.sqrt(self.MSE)

    @abstractmethod
    def loss(self):
        pass
