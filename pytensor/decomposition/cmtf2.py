import h5py
import numpy as np
from .cp import CP_ALS

from . import decompositions
from ..base import unfold
from .. import base

class CMTF_ALS(CP_ALS):
    DecompositionType = decompositions.CoupledTensors
    @property
    def SSE(self):
        """Sum Squared Error"""
        # TODO: Cache result
        return np.linalg.norm(self.X - self.reconstructed_X)**2 + self.coupled_factor_matrices_SSE

    @property
    def MSE(self):
        #raise NotImplementedError('Not implemented') 
        # TODO: fix this
        num_elements = np.prod(self.X.shape) + sum(np.prod(Yi.shape) for Yi in self.coupled_factor_matrices)
        return self.SSE/num_elements

    @property
    def coupled_factor_matrices_SSE(self):
        SSE = 0

        for Y, reconstructed_Y in zip(self.coupled_matrices, self.reconstructed_coupled_matrices):
            SSE += np.linalg.norm(Y - reconstructed_Y)**2
        return SSE

    @property
    def RMSE(self):
        return np.sqrt(self.MSE)

    @property
    def reconstructed_coupled_matrices(self):
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

    def fit_transform(self, X, coupled_matrices, coupling_modes, y=None, max_its=None, tensor_missing_values=None, impute_matrix_axis=None):
        self.fit(X=X, coupled_matrices=coupled_matrices, coupling_modes=coupling_modes, y=y, max_its=max_its, tensor_missing_values=tensor_missing_values, impute_matrix_axis=impute_matrix_axis)
        return self.decomposition

    def fit(self, X, coupled_matrices, coupling_modes, y, max_its=None, tensor_missing_values=None, impute_matrix_axis=None):
        self._init_fit(X=X, coupled_matrices=coupled_matrices, coupling_modes=coupling_modes, initial_decomposition=None, tensor_missing_values=tensor_missing_values, impute_matrix_axis=impute_matrix_axis)
        super()._fit()

    def init_random(self):
        """Random initialisation of the factor matrices.

        Each element of the factor matrices are taken from a standard normal distribution.
        """
        pass

    def _init_fit(self, X, coupled_matrices, coupling_modes, initial_decomposition=None, max_its=None, tensor_missing_values=None, impute_matrix_axis=None):
        self.decomposition = self.DecompositionType.random_init(tensor_sizes=X.shape, rank=self.rank,
            matrices_sizes=[mat.shape for mat in coupled_matrices],coupling_modes=coupling_modes)
        self.coupled_matrices = coupled_matrices
        super()._init_fit(X=X, max_its=max_its, initial_decomposition=initial_decomposition, missing_values=tensor_missing_values)
        if impute_matrix_axis is not None:
        #    mats_with_missing  = [mat for mat in coupled_matrices if np.isnan(mat).any()]
            self.Ns = [np.ones(mat.shape) for mat in self.coupled_matrices]
            for i, N in enumerate(self.Ns):
                inds = np.where(np.isnan(self.coupled_matrices[i]))
                self.Ns[i][inds] = 0
            self._init_impute_matrices_missing(impute_matrix_axis)
    
    def _init_impute_matrices_missing(self, axis):
        for i, mat in enumerate(self.coupled_matrices):
            n = 0
            if np.isnan(mat).any():
                axis_means = np.nanmean(mat, axis=axis[n])    
                inds = np.where(np.isnan(mat))  
                self.coupled_matrices[i][inds] = np.take(axis_means, inds[1 if axis[n]==0 else 0])
                n+=1

    def _set_new_matrices(self):
        for i, N in enumerate(self.Ns):
            self.coupled_matrices[i] = self.coupled_matrices[i] * N + self.reconstructed_coupled_matrices[i] * (np.ones(shape=N.shape) - N)

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

    def _update_uncoupled_matrix_factors(self):
        for i, mode in enumerate(self.coupling_modes):
            lhs = self.factor_matrices[mode].T
            rhs = self.coupled_matrices[i].T

            if self.non_negativity_constraints is None:
                self.uncoupled_factor_matrices[i][...] = base.rightsolve(lhs, rhs)

            if self.non_negativity_constraints[mode]:
                new_fm = base.non_negative_rightsolve(lhs, rhs)
                self.uncoupled_factor_matrices[i][...] = new_fm
            else:
                self.uncoupled_factor_matrices[i][...] = base.rightsolve(lhs, rhs)

