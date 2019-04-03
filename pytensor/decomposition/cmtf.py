import h5py
import numpy as np
from .cp import CP_ALS

from ..base import unfold
from .. import base



class CMTF_ALS(CP_ALS): 

    def store_checkpoint(self):
        #TODO: store the matrices as well
        with h5py.File(self.checkpoint_path, 'a') as h5:
            h5.attrs['final_iteration'] = self.current_iteration
            checkpoint_group = h5.create_group(f'checkpoint_{self.current_iteration:05d}')
            self.decomposition.store_in_hdf5_group(checkpoint_group)
            for logger in self.loggers:
                logger.write_to_hdf5_group(h5)


    @property
    def coupled_factor_matrices(self):
        return [self.factor_matrices[mode] for mode in self.coupling_modes]
            
    @property
    def reconstructed_coupled_matrices(self):
        reconstructed = []
        for scores, loadings, weights in zip(self.coupled_factor_matrices, self.uncoupled_factor_matrices, self.coupled_weights):
            reconstructed.append(scores @ loadings.T)
        return reconstructed

    @property
    def coupled_factor_matrices_SSE(self):
        SSE = 0

        for Y, reconstructed_Y in zip(self.coupled_matrices, self.reconstructed_coupled_matrices):
            SSE += np.linalg.norm(Y - reconstructed_Y)**2
        return SSE
    
    @property
    def SSE(self):
        """Sum Squared Error"""
        # TODO: Cache result
        return np.linalg.norm(self.X - self.reconstructed_X)**2 + self.coupled_factor_matrices_SSE

    @property
    def MSE(self):
        #raise NotImplementedError('Not implemented') 
        # TODO: fix this
        return self.SSE

    @property
    def RMSE(self):
        pass
        raise NotImplementedError('Not implemented') 

    def loss(self):
        return self.SSE #TODO: skal det være property?

    def set_coupled_matrices(self, coupled_matrices, coupling_modes):
        self.coupled_matrices = coupled_matrices
        self.coupling_modes = coupling_modes
        self._set_mode_to_coupled_matrix_mapping(coupling_modes)

    def _set_mode_to_coupled_matrix_mapping(self, coupling_modes):
        mode_to_cm_idx = {}

        for i, cm in enumerate(coupling_modes):
            if cm in mode_to_cm_idx:
                mode_to_cm_idx[cm].append(i)
            else:
                mode_to_cm_idx[cm] = [i]
        self.mode_to_cm_idx = mode_to_cm_idx
    
    def _init_fit(self, X, coupled_matrices, coupling_modes, max_its, initial_decomposition):
        self.set_coupled_matrices(coupled_matrices, coupling_modes)
        super()._init_fit(X=X, max_its=max_its, initial_decomposition=initial_decomposition)
        self._rel_function_change = np.inf
        self.prev_SSE = self.SSE
        
        self.decomposition.normalize_components()


    def fit(self, X, coupled_matrices, coupling_modes, y=None, *, max_its=None, initial_decomposition=None):
        """Fit a CMTF model. Precomputed components must be specified if init method is `precomputed`.

        Arguments:
        ----------
        X : np.ndarray
            The tensor to fit
        coupled_matrices: list(np.ndarray)
            Matrices to fit
        coupling_modes: list(int)
            Modes to couple along
        y : None
            Ignored, included to follow sklearn standards.
        max_its : int (optional)
            If set, then this will override the class's max_its.
        initial_decomposition : pytensor.base.KruskalTensor or str
            The initial KruskalTensor (init=precomputed) to use or the path of the 
            logfile to load (init=from_file).
        """
        self._init_fit(
            X=X, coupled_matrices=coupled_matrices, coupling_modes=coupling_modes, max_its=max_its, 
            initial_decomposition=initial_decomposition
        )
        self._fit()

    def fit_transform(self, X, coupled_matrices, coupling_modes, y=None, *, max_its=None, initial_decomposition=None):
        """Fit a CMTF model and return kruskal tensor together with decomposed matrices. 
        
        Precomputed components must be specified if init method is `precomputed`.

        Arguments:
        ----------
        X : np.ndarray
            The tensor to fit
        coupled_matrices: list(np.ndarray)
            Matrices to fit
        coupling_modes: list(int)
            Modes to couple along
        y : None
            Ignored, included to follow sklearn standards.
        max_its : int (optional)
            If set, then this will override the class's max_its.
        initial_decomposition : tuple
            A tuple parametrising a Kruskal tensor.
            The first element is a list of factor matrices and the second element is an array containing the weights.
        """
        self.fit(X=X, coupled_matrices=coupled_matrices, coupling_modes=coupling_modes,
                 y=y, max_its=max_its, initial_decomposition=initial_decomposition)
        return self.decomposition, [[A,V] for A, V in zip(self.coupled_factor_matrices, self.uncoupled_factor_matrices)]

    def _update_als_factors_(self):
        super()._update_als_factors()
        self._update_uncoupled_matrix_factors()

    def _get_als_lhs(self, mode):
        if mode in self.mode_to_cm_idx:
            khatri_rao_product= base.khatri_rao(*self.factor_matrices, skip=mode)
            V = [self.uncoupled_factor_matrices[cm_idx] for cm_idx in self.mode_to_cm_idx[mode]][0]
            return np.concatenate([khatri_rao_product, V], axis=0).T
        else:
            return self._compute_V(mode).T
    
    def _get_als_rhs(self, mode):
        if mode in self.mode_to_cm_idx:
            unfolded_X = base.unfold(self.X, mode)
            coupled_Y = [self.coupled_matrices[cm_idx] for cm_idx in self.mode_to_cm_idx[mode]][0]

            return np.concatenate([unfolded_X, coupled_Y], axis=1)
        else:
            return base.matrix_khatri_rao_product(self.X, self.factor_matrices, mode)

    def _update_als_factor_(self, mode):
        """Solve least squares problem to get factor for one mode."""
        debug = True
        if debug:
            from textwrap import dedent
            debugstring = dedent(f"""\
            
            
            Beginning of _update_als_factor mode {mode}

                iteration: {self.current_iteration}
                V weigths: {self.coupled_weights}
                weights: {self.weights}
                loss: {self.loss()}
                tensor loss: {np.linalg.norm(self.X - self.reconstructed_X)**2}
                Y loss {self.coupled_factor_matrices_SSE}
            """)
            _loss = self.loss()
        _lhs = self._get_als_lhs(mode)
        _rhs = self._get_als_rhs(mode)

        new_factor = _rhs@np.linalg.pinv(_lhs)
        
        self.decomposition.normalize_components()

        self.factor_matrices[mode][...] = new_factor

        if debug:
            debugstring += dedent(f"""\
            end of update als factor (mode {mode})
                loss: {self.loss()})
                tensor loss: {np.linalg.norm(self.X - self.reconstructed_X)**2})
                Y loss: {self.coupled_factor_matrices_SSE}""")

            if self.loss() > _loss:
                print(debugstring)

    def _update_uncoupled_matrix_factors(self):
        for mode, cm_idx in self.mode_to_cm_idx.items():
            debug = True
            if debug:
                from textwrap import dedent
                debugstring = dedent(f"""\
                
                
                Beginning of uncoupled mode {mode}

                    iteration: {self.current_iteration}
                    V weigths: {self.coupled_weights}
                    weights: {self.weights}
                    loss: {self.loss()}
                    tensor loss: {np.linalg.norm(self.X - self.reconstructed_X)**2}
                    Y loss {self.coupled_factor_matrices_SSE}
                """)
                _loss = self.loss()
            unfolded_X = base.unfold(self.X, mode)
            khatri_rao_product= base.khatri_rao(*self.factor_matrices, skip=mode)
            cm_idx = self.mode_to_cm_idx[mode][0]
            
            coupled_Y = [self.coupled_matrices[cm_idx] for cm_idx in self.mode_to_cm_idx[mode]][0].T
            V = [self.uncoupled_factor_matrices[cm_idx] for cm_idx in self.mode_to_cm_idx[mode]][0].T

            self.uncoupled_factor_matrices[cm_idx][...] = coupled_Y @ np.linalg.pinv(self.factor_matrices[mode]).T
            if debug:
                debugstring += dedent(f"""\
                end of update als factor (mode {mode})
                    loss: {self.loss()})
                    tensor loss: {np.linalg.norm(self.X - self.reconstructed_X)**2})
                    Y loss: {self.coupled_factor_matrices_SSE}""")

                if self.loss() > _loss:
                    print(debugstring)



    def _update_als_factor(self, mode):
        """Solve least squares problem to get factor for one mode."""

        """        
        for w in self.coupled_weights:
            print('V weights:', w)

        for w in self.weights:
            print('factor weights:', w)
        """

        debug = True
        if debug:
            from textwrap import dedent
            debugstring = dedent(f"""\
            
            
            Beginning of _update_als_factor mode {mode}

                iteration: {self.current_iteration}
                V weigths: {self.coupled_weights}
                weights: {self.weights}
                loss: {self.loss()}
                tensor loss: {np.linalg.norm(self.X - self.reconstructed_X)**2}
                Y loss {self.coupled_factor_matrices_SSE}
            """)
            _loss = self.loss()
        



        unfolded_X = base.unfold(self.X, mode)
        khatri_rao_product= base.khatri_rao(*self.factor_matrices, skip=mode)


        if mode in self.mode_to_cm_idx:

            cm_idx = self.mode_to_cm_idx[mode][0]
            
            #self.coupled_weights[cm_idx][...] = np.linalg.norm(self.uncoupled_factor_matrices[cm_idx], axis=0)
            #self.uncoupled_factor_matrices[cm_idx][...] = self.uncoupled_factor_matrices[cm_idx]/self.coupled_weights[cm_idx][np.newaxis]

            # TODO: support multiple couplings for one mode
            coupled_Y = [self.coupled_matrices[cm_idx] for cm_idx in self.mode_to_cm_idx[mode]][0]

            concat_X = np.concatenate([unfolded_X, coupled_Y], axis=1)
            V = [self.uncoupled_factor_matrices[cm_idx] for cm_idx in self.mode_to_cm_idx[mode]][0]
            concat_b = np.concatenate([khatri_rao_product, V], axis=0).T 


            new_factor = concat_X @ np.linalg.pinv(concat_b)

            self.factor_matrices[mode][...] = new_factor

            self.factor_matrices[mode][...] = self.factor_matrices[mode][...]/np.linalg.norm(self.factor_matrices[mode], axis=0)



            #self.coupled_matrices[cm_idx][...] =  coupled_Y @ np.linalg.pinv(self.uncoupled_factor_matrices[mode]).T
            self.uncoupled_factor_matrices[cm_idx][...] = coupled_Y.T @ np.linalg.pinv(self.factor_matrices[mode]).T
        else:
            """
            concat_X = unfolded_X
            concat_b = khatri_rao_product.T

            new_factor = concat_X @ np.linalg.pinv(concat_b)

            self.factor_matrices[mode][...] = new_factor
            """
            V = self._compute_V(mode)
            n = np.prod(V.shape)

            _rhs = base.matrix_khatri_rao_product(self.X, self.factor_matrices, mode).T
            # U, S, W = np.linalg.svd(V.T, full_matrices=False)
            # new_factor = (W.T @ np.diag(1/(S + 1e-5)) @ U.T @ _rhs).T  #TODO er det denne måten vi vil gjøre det?
            new_factor = (np.linalg.pinv(V.T) @ _rhs).T
            new_factor = _rhs.T@np.linalg.pinv(V)

            self.factor_matrices[mode][...] = new_factor
            #self.decomposition.normalize_components()

        
        if debug:
            debugstring += dedent(f"""\
            end of update als factor (mode {mode})
                loss: {self.loss()})
                tensor loss: {np.linalg.norm(self.X - self.reconstructed_X)**2})
                Y loss: {self.coupled_factor_matrices_SSE}""")

            if self.loss() > _loss:
                print(debugstring)
            
        
    
    def _init_coupled_matrices(self):

        self.uncoupled_factor_matrices = [None]*len(self.coupling_modes)
        self.coupled_weights = [None]*len(self.coupling_modes)

        for i, mode in enumerate(self.coupling_modes):

            num_rows = self.coupled_matrices[i].shape[1]
            num_columns = self.rank
            self.uncoupled_factor_matrices[i] = np.random.randn(num_rows, num_columns)
            self.uncoupled_factor_matrices[i][...] = self.uncoupled_factor_matrices[i]/np.linalg.norm(self.uncoupled_factor_matrices[i], axis=0)
            self.coupled_weights[i] = np.ones((self.rank,))

    def init_components(self, initial_decomposition=None):
        
        super().init_components(initial_decomposition=initial_decomposition)
        self._init_coupled_matrices()