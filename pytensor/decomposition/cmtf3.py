import h5py
import numpy as np
from .cp import CP_ALS
import itertools

from . import decompositions
from ..base import unfold
from .. import base

class CMTF_ALS(CP_ALS):
    """Coupled Tensor decomposition using Alternating Least Squares.
    """
    DecompositionType = decompositions.CoupledTensors2
    @property
    def SSE(self):
        """Computes the sum squared error of the decomposition
        
        Returns
        -------
        float
            Sum of squared error.
        """
        # TODO: Cache result
        return np.linalg.norm(self.X - self.reconstructed_X)**2 + self.coupled_tensors_SSE

    @property
    def MSE(self):
        """Computes the mean squared error of the decomposition.
        
        Returns
        -------
        float
            Mean of squared error.
        """
        #raise NotImplementedError('Not implemented') 
        # TODO: fix this
        num_elements = np.prod(self.X.shape) + sum(np.prod(Yi.shape) for Yi in self.original_tensors)
        return self.SSE/num_elements

    @property
    def coupled_tensors_SSE(self):
        """Computes total SSE for all coupled matrices.
        
        Returns
        -------
        float
            SSE for couple matrices.
        """
        SSE = 0

        for Y, reconstructed_Y in zip(self.original_tensors, self.reconstructed_coupled_tensors):
            SSE += np.linalg.norm(Y - reconstructed_Y)**2
        return SSE

    @property
    def RMSE(self):
        """        
        Returns RMSE of the decomposition
        """
        return np.sqrt(self.MSE)

    @property
    def reconstructed_coupled_tensors(self):
        """Constructs the coupeld tensors from decomposition.        
        Returns
        -------
        list(np.ndarray)
            The coupled tensosr.
        """
        return self.decomposition.construct_coupled_tensors()

    @property
    def coupled_factor_matrices(self):
        """The coupled factor matrices for the the coupled tensors.
        
        Returns
        -------
        list(np.ndarray)
            The coupled factor matrices.
        """
        return self.decomposition.coupled_factor_matrices

    @property
    def uncoupled_tensor_factors(self):
        """The uncoupeld factor matrices for the coupled tensors.

        Returns
        -------
        list(np.ndarray)
            The uncoupled factor matrices.
        """
        return self.decomposition.uncoupled_tensor_factors
    
    @property
    def num_coupled_tensors(self):
        """Number of coupled tensors that are not matrices, i.e. dim > 2.
        
        Returns
        -------
        int
            Number of coupled tensors.
        """
        return len(self.decomposition.coupled_tensors)

    @property
    def coupling_modes(self):
        """        
        Returns
        -------
        list(int)
            The modes the coupled tensors are coupled to the main tensor along.
        """
        return self.decomposition.coupling_modes

    def fit_transform(self, X, coupled_tensors, coupling_modes, y=None, max_its=None, tensor_missing_values=None, coupled_missing_values=None, penalty=None):
        """Executes coupled-tensor-matrix factorisation and returns the decomposition
        
        Parameters
        ----------
        X : np.ndarray
            The n-dimensional tensor to fit, n>2.
        coupled_tensors : list(np.ndarray)
            The coupled tensors to fit. All tensors of dimension 2 (matrices) must be at end of list.
        coupling_modes : list(int)
            Modes to couple along, must be ordered like coupled_tensors.
        y : None
            Ignored, included to follow sklearn standards.
        max_its : int, optional
            If set, this will override the class's max_its
        tensor_missing_values : int or tuple(np.ndarray), optional
            Use if main tensor has Nan-values. If int, imputes mean along the given axis.
            If tuple(np.ndarray), assumes these indices to be unknown but pre-imputed.
        coupled_missing_values : lis(int or None), optional
            Use if coupled tensors have NaNs. Must be same length and ordered as coupled_tensors. 
            Takes values in list and imputes means along the axis.
        penalty: float, optional
            The penalty for L1-regularisation of the weights. Use if using ACMTF.
        
        Returns
        -------
        decompositions.CoupledTensors
            The decomposed main and coupeld tensors.
        """
        self.fit(X=X, coupled_tensors=coupled_tensors, coupling_modes=coupling_modes, y=y, max_its=max_its, tensor_missing_values=tensor_missing_values, coupled_missing_values=coupled_missing_values, penalty=penalty)
        return self.decomposition

    def fit(self, X, coupled_tensors, coupling_modes, y, max_its=None, tensor_missing_values=None, coupled_missing_values=None, penalty=None):
        """Fits a CMTF model. 
        
        Parameters
        ----------
        X : np.ndarray
            The n-dimensional tensor to fit, n>2.
        coupled_tensors : list(np.ndarray)
            The coupled tensors to fit. All tensors of dimension 2 (matrices) must be at end of list.
        coupling_modes : list(int)
            Modes to couple along, must be ordered like coupled_tensors.
        y : None
            Ignored, included to follow sklearn standards.
        max_its : int, optional
            If set, this will override the class's max_its
        tensor_missing_values : int or tuple(np.ndarray), optional
            Use if main tensor has Nan-values. If int, imputes mean along the given axis.
            If tuple(np.ndarray), assumes these indices to be unknown but pre-imputed.
        coupled_missing_values : lis(int or None), optional
            Use if coupled tensors have NaNs. Must be same length and ordered as coupled_tensors. 
            Takes values in list and imputes means along the axis.
        penalty: float, optional
            The penalty for L1-regularisation of the weights. Use if using ACMTF.
        """
        self._init_fit(X=X, coupled_tensors=coupled_tensors, coupling_modes=coupling_modes, initial_decomposition=None, tensor_missing_values=tensor_missing_values, coupled_missing_values=coupled_missing_values, penalty=penalty)
        super()._fit()

    def init_random(self):
        """Dummy function.
        """
        pass

    def _init_fit(self, X, coupled_tensors, coupling_modes, initial_decomposition=None, max_its=None, tensor_missing_values=None, coupled_missing_values=None, penalty=None):
        """Initialises the factorisation.
        
        Parameters
        ----------
        X : np.ndarray
            The n-dimensional tensor to fit, n>2.
        coupled_tensors : list(np.ndarray)
            The coupled tensors to fit. All tensors of dimension 2 (matrices) must be at end of list.
        coupling_modes : list(int)
            Modes to couple along, must be ordered like coupled_tensors.
        initial_decomposition : None
            Not implemented.
        max_its : int, optional
            If set, this will override the class's max_its
        tensor_missing_values : int or tuple(np.ndarray), optional
            Use if main tensor has Nan-values. If int, imputes mean along the given axis.
            If tuple(np.ndarray), assumes these indices to be unknown but pre-imputed.
        coupled_missing_values : lis(int or None), optional
            Use if coupled tensors have NaNs. Must be same length and ordered as coupled_tensors. 
            Takes values in list and imputes means along the axis.
        penalty: float, optional
            The penalty for L1-regularisation of the weights. Use if using ACMTF.
        """
        #TODO: if-test to check that tensors and matrices are ordered [tensors, matrices]
        self.penalty = penalty
        self.missing = True if coupled_missing_values is not None else False
        self.decomposition = self.DecompositionType.random_init(main_tensor_shape=X.shape, rank=self.rank,
            coupled_tensors_shapes=[tensor.shape for tensor in coupled_tensors],coupling_modes=coupling_modes)
        self.original_tensors = coupled_tensors
        super()._init_fit(X=X, max_its=max_its, initial_decomposition=initial_decomposition, missing_values=tensor_missing_values)
        if coupled_missing_values is not None:
        #    mats_with_missing  = [mat for mat in coupled_matrices if np.isnan(mat).any()]
            self.Ms = [np.ones(tensor.shape) for tensor in coupled_tensors[:self.num_coupled_tensors]]
            for i, M in enumerate(self.Ms):
                inds = np.where(np.isnan(coupled_tensors[i]))
                self.Ms[i][inds] = 0
            self.Ns = [np.ones(mat.shape) for mat in coupled_tensors[self.num_coupled_tensors:]]
            for i, N in enumerate(self.Ns):
                inds = np.where(np.isnan(coupled_tensors[i+self.num_coupled_tensors]))
                self.Ns[i][inds] = 0
            self._init_impute_tensors_missing(coupled_missing_values)
    
    def _init_impute_tensors_missing(self, axes):
        """Mean-imputes coupled tensors with missing values (np.nan).
        
        Parameters
        ----------
        axes : list(int)
            The axes to impute a long. 
        
        Raises
        ------
        Exception
            If the number of axes is different to the number of coupled matrices.
        ValueError
            If the list of axes contains different values from 0, 1 or None.
        """
        if len(axes) != len(self.original_tensors):
            raise Exception("Number of tensors and axes must be the same."
                            " Got {0} matrices and {1} axes. Axes must be list of 0 or 1 or None."
                            .format(len(self.original_tensors), len(axes)))
        if not(all(ax == 0 or ax==1 or ax is None for ax in axes)):
                raise ValueError("Axes to impute along must all be either 0, 1 or None.")
        tensor_axes = axes[:self.num_coupled_tensors]
        matrix_axes = axes[self.num_coupled_tensors:]

        for ind, axis in enumerate(tensor_axes):
            C = np.copy(self.original_tensors[ind])
            nan_locs = np.where(np.isnan(C))
            for i, j, k in zip(nan_locs[0], nan_locs[1], nan_locs[2]):
                if axis == 0:
                    C[i, j, k] = np.nanmean(self.original_tensors[ind][:, j, k])
                elif axis == 1:
                    C[i, j, k] = np.nanmean(self.original_tensors[ind][i,:, k])
                elif axis == 2:
                    C[i, j, k] = np.nanmean(self.original_tensors[ind][i, j, :])
            self.original_tensors[ind] = np.copy(C)

        for i, axis in enumerate(matrix_axes):
            if axis is None:
                continue
            axis_means = np.nanmean(self.original_tensors[i+self.num_coupled_tensors], axis=axis)    
            inds = np.where(np.isnan(self.original_tensors[i+self.num_coupled_tensors]))  
            self.original_tensors[i+self.num_coupled_tensors][inds] = np.take(axis_means, inds[0 if axis==1 else 1])

    def _set_new_tensors(self):
        """Updates all coupled tensors. A tensor is only changed if it originally had missing values.
        """
        for i, M in enumerate(self.Ms):
            self.original_tensors[i] = self.original_tensors[i] * M + self.reconstructed_coupled_tensors[i] * (np.ones(shape=M.shape) - M)
        for i, N in enumerate(self.Ns):
            ind = i + self.num_coupled_tensors
            self.original_tensors[ind] = self.original_tensors[ind] * N + self.reconstructed_coupled_tensors[ind] * (np.ones(shape=N.shape) - N)

    def _update_als_factors(self):
        """Updates factors with alternating least squares.
        """
        num_modes = len(self.X.shape) # TODO: Should this be cashed?
        for mode in range(num_modes):
            if self.non_negativity_constraints[mode]:
                self._update_als_factor_non_negative(mode) 
            else:
                self._update_als_factor(mode)
        self._update_uncoupled_tensor_factors()
        if self.missing:
            self._set_new_tensors()
        if self.penalty:
            self._reguralize_weights()
        
    def _reguralize_weights(self):
        """Reguralises the weights of the decomposition with L1.
        """
        self.decomposition.reset_weights()
        self.decomposition.normalize_components()
        for ind, tensor in enumerate([self.decomposition.main_tensor] +self.decomposition.coupled_tensors):
            weights = tensor.weights
            A, B, C = tensor.factor_matrices
            l = np.zeros(self.rank)
            top = np.zeros(self.rank)
            bot = np.zeros(self.rank)
            ranks = np.arange(0, self.rank)
            for r in ranks:
                for i, j, k in itertools.product(range(A.shape[0]), range(B.shape[0]), range(C.shape[0])):
                    #bot[r] += (A[i, r]*B[j, r]*C[k, r])**2
                    if ind == 0:
                        top[r] += A[i, r]*B[j, r]*C[k, r] * (self.X[i, j, k] - sum([weights[rank]*A[i, rank]*B[j, rank]*C[k, rank] for rank in ranks if rank != r]))
                    else:
                        top[r] += A[i, r]*B[j, r]*C[k, r] * (self.original_tensors[ind-1][i, j, k] - sum([weights[rank]*A[i, rank]*B[j, rank]*C[k, rank] for rank in ranks if rank != r]))
                if np.isclose(weights[r], 0):
                    l[r] = top[r] / bot[r] 
                else:
                    #TODO: should it be .5*penalty? does it matter?
                    l[r] = (top[r] + self.penalty * (1 if abs(weights[r])>0 else -1)) #/ bot[r] 
            tensor.weights[...] = l
        for ind, mat in enumerate(self.decomposition.coupled_matrices):
            weights = mat.weights
            A, V = mat.factor_matrices
            s = np.zeros(self.rank)
            top = np.zeros(self.rank)
            bot = np.zeros(self.rank)
            for r in ranks:
                for i, j in itertools.product(range(A.shape[0]), range(V.shape[0])):
                    top[r] += A[i, r]*V[j, r] * (self.original_tensors[self.num_coupled_tensors+ind][i, j] - sum([weights[rank]*A[i, rank]*V[j, rank] for rank in ranks if rank != r]))
                    #bot[r] += (A[i, r]*V[j, r])**2

                if np.isclose(weights[r], 0):
                    print('heyisclose', weights[r])
                    s[r] = top[r] #/ bot[r] 
                else:
                #TODO: should it be .5*penalty? does it matter?
                    s[r] = (top[r] - self.penalty * (1 if abs(weights[r])>0 else -1))# / bot[r] 
            mat.weights = s

    def _update_als_factor(self, mode):
        """Solve least squares problem to get factor for one mode.
        """
        lhs = self._get_als_lhs(mode)
        rhs = self._get_als_rhs(mode)

        rightsolve = self._get_rightsolve(mode)

        new_factor = rightsolve(lhs, rhs)
        self.factor_matrices[mode][...] = new_factor

    def _get_als_lhs(self, mode):
        """Compute left hand side of least squares problem.
        """
        # TODO: make this nicer, make a self.mat_coupling_modes?
        if mode in self.coupling_modes:
            factors = [np.copy(mat) for mat in self.factor_matrices]
            if mode != 0:
                factors[0] = self.decomposition.main_tensor.weights*factors[0]
            else:
                factors[1] = self.decomposition.main_tensor.weights*factors[1]
            khatri_rao_products = base.khatri_rao(*factors, skip=mode)
            for i, tensor in enumerate(self.decomposition.coupled_tensors):
                if self.coupling_modes[i] != mode:
                    continue
                factors = [np.copy(mat) for mat in tensor.factor_matrices]
                if mode != 0:
                    factors[0] = tensor.weights*factors[0]
                else:
                    factors[1] = tensor.weights*factors[1]
                khatri_rao_products = np.concatenate([khatri_rao_products, base.khatri_rao(*factors, skip=mode)])  

            #Checking whether there are any matrices coupled on the current mode
            mat_couplings = self.coupling_modes[self.num_coupled_tensors:]
            n_couplings = mat_couplings.count(mode)
            if n_couplings > 0:    
                
                matrix_factors = [np.copy(mat.factor_matrices[1]) for mat in self.decomposition.coupled_matrices]
                indices = [i for i, cplmode in enumerate(mat_couplings) if cplmode == mode]
                weights = [mat.weights for mat in self.decomposition.coupled_matrices]
                V = weights[indices[0]] * self.uncoupled_tensor_factors[self.num_coupled_tensors+indices[0]]
                if  n_couplings > 1:
                    for i in indices[1:]:
                        V = np.concatenate([V, weights[i]*matrix_factors[i]], axis=0)
                return np.concatenate([khatri_rao_products, V], axis=0).T
            else:
                return khatri_rao_products.T
        else:
            # V = np.ones((self.rank, self.rank))
            # TODO: this was a problem, dunno why
            # for i, factor in enumerate(self.factor_matrices):
            #     if i == mode:
            #         continue
            #     V *= (self.decomposition.tensor.weights*factor).T @ factor
            # return V
            factors = [np.copy(mat) for mat in self.factor_matrices]
            if mode != 0:
                factors[0] = self.decomposition.main_tensor.weights*factors[0]
            else:
                factors[1] = self.decomposition.main_tensor.weights*factors[1]
            return base.khatri_rao(*factors, skip=mode).T
             
    
    def _get_als_rhs(self, mode):
        """Compute right hand side of least squares problem.
        """
        if mode in self.coupling_modes:
            rhs = base.unfold(self.X, mode)
            #Original_tensors needs to be sorted with all coupled tensors first, then coupled matrices
            for i, tensor in enumerate(self.original_tensors):
                if self.coupling_modes[i] == mode:
                    if len(tensor.shape) == 2:
                        rhs = np.concatenate([rhs, tensor], axis=1)
                    else:
                        rhs = np.concatenate([rhs, base.unfold(tensor, mode)], axis=1)
            return rhs
        else:
            # TODO: this was a problem, dunno why
            # factors = [self.decomposition.tensor.weights * mat for mat in self.factor_matrices]
            # return base.matrix_khatri_rao_product(self.X, factors, mode)
            return base.unfold(self.X, mode)

    def _update_uncoupled_tensor_factors(self):
        """Solve ALS problem for uncoupled tensor factors.
        """
        for i, tensor in enumerate(self.decomposition.coupled_tensors):
            coupl_mode = self.coupling_modes[i]
            modes = np.arange(0, len(tensor.shape))
            modes = modes[modes != coupl_mode]
            for mode in modes:
                lhs = np.ones((self.rank, self.rank))
                for j, factor in enumerate(tensor.factor_matrices):
                    if j == mode:
                        continue
                    lhs *= factor.T @ factor
                rhs = base.matrix_khatri_rao_product(self.original_tensors[i], tensor.factor_matrices, mode)

                if self.non_negativity_constraints is None:
                    tensor.factor_matrices[i][...] = base.rightsolve(lhs, rhs)

                if self.non_negativity_constraints[mode]:
                    new_fm = base.non_negative_rightsolve(lhs, rhs)
                    tensor.factor_matrices[i][...] = new_fm
                else:
                    tensor.factor_matrices[mode][...] = base.rightsolve(lhs, rhs)
        
        for i, mode in enumerate(self.coupling_modes[self.num_coupled_tensors:]):
            lhs = (self.decomposition.coupled_matrices[i].weights*self.factor_matrices[mode]).T
            rhs = self.original_tensors[self.num_coupled_tensors + i].T

            if self.non_negativity_constraints is None:
                self.decomposition.coupled_matrices[i].factor_matrices[1][...] = base.rightsolve(lhs, rhs)

            if self.non_negativity_constraints[mode]:
                new_fm = base.non_negative_rightsolve(lhs, rhs)
                self.decomposition.coupled_matrices[i].factor_matrices[1][...] = new_fm
            else:
                self.decomposition.coupled_matrices[i].factor_matrices[1][...] = base.rightsolve(lhs, rhs)