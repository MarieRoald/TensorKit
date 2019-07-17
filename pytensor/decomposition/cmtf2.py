import h5py
import numpy as np
from .cp import CP_ALS

from . import decompositions
from ..base import unfold
from .. import base

class CMTF_ALS(CP_ALS):
    DecompositionType = decompositions.CoupledTensors

    @property
    def coupled_matrices(self):
        return self.decomposition.construct_matrices()

    @property
    def coupled_factor_matrices(self):
        return self.decomposition.coupled_factor_matrices

    @property
    def uncoupled_factor_matrices(self):
        return self.decomposition.uncoupled_factor_matrices
    
    @property
    def coupling_modes(self):
        return self.decomposition.coupling_modes

    def fit_transform(self, X, coupled_matrices, coupling_modes, y=None, max_its=None):
        self.fit(X=X, coupled_matrices=coupled_matrices, coupling_modes=coupling_modes, y=y, max_its=max_its)
        
    def fit(self, X, coupled_matrices, coupling_modes, y, max_its=None):
        self._init_fit(X=X, coupled_matrices=coupled_matrices, coupling_modes=coupling_modes, initial_decomposition=None)

    def init_random(self):
        """Random initialisation of the factor matrices.

        Each element of the factor matrices are taken from a standard normal distribution.
        """
        pass

    def _init_fit(self, X, coupled_matrices, coupling_modes, initial_decomposition=None, max_its=None):
        self.decomposition = self.DecompositionType.random_init(tensor_sizes=X.shape, rank=self.rank,
            matrices_sizes=[mat.shape for mat in coupled_matrices],coupling_modes=coupling_modes)
        self.couplings = coupled_matrices
        super()._init_fit(X=X, max_its=max_its, initial_decomposition=initial_decomposition)
        self._rel_function_change = np.inf
        self.prev_SSE = self.SSE
        
    def _update_als_factors(self):
        num_modes = len(self.X.shape) # TODO: Should this be cashed?
        for mode in range(num_modes):
            if self.non_negativity_constraints[mode]:
                self._update_als_factor_non_negative(mode) 
            else:
                self._update_als_factor(mode)
        self._update_uncoupled_matrix_factors()

    def _update_als_factor(self, mode):
        """Solve least squares problem to get factor for one mode."""
        lhs = self._get_als_lhs(mode)
        rhs = self._get_als_rhs(mode)

        rightsolve = self._get_rightsolve(mode)

        new_factor = rightsolve(lhs, rhs)
        self.factor_matrices[mode][...] = new_factor

    def _get_als_lhs(self, mode):
        """Compute left hand side of least squares problem."""
        # TODO: make this nicer.
        if mode in self.coupling_modes:
            
            n_couplings = self.coupling_modes.count(mode)
            khatri_rao_product = base.khatri_rao(*self.factor_matrices, skip=mode)
            indices = [i for i, cplmode in enumerate(self.coupling_modes) if cplmode == mode]
            V = self.uncoupled_factor_matrices[indices[0]]
            if  n_couplings > 1:
                for i in indices[1:]:
                    V = np.concatenate([V, self.uncoupled_factor_matrices[indices[i]]], axis=0)
            return np.concatenate([khatri_rao_product, V], axis=0).T
        else:
            return super()._get_als_lhs(mode)
    
    def _get_als_rhs(self, mode):
        if mode in self.coupling_modes:
            unfolded_X = base.unfold(self.X, mode)
            n_couplings = self.coupling_modes.count(mode)
            indices = [i for i, cplmode in enumerate(self.coupling_modes) if cplmode == mode]
            
            coupled_Y = self.coupled_matrices[indices[0]]
            if  n_couplings > 1:              
                for i in indices[1:]:
                    coupled_Y = np.concatenate([coupled_Y,
                     self.coupled_matrices[indices[i]]], axis=1)
            return np.concatenate([unfolded_X, coupled_Y], axis=1)
        else:
            return super()._get_als_rhs(mode)