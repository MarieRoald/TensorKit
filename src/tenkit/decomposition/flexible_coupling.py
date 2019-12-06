from copy import copy
from pathlib import Path

import h5py
import numpy as np

from .. import base, utils
from . import decompositions
from .base_decomposer import BaseDecomposer
from .cp import CP_ALS
from .parafac2 import Parafac2_ALS
from .decompositions import Parafac2Tensor
from .utils import quadratic_form_trace


class BaseCoupledMatrices(BaseDecomposer):
    DecompositionType = decompositions.EvolvingTensor
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
        tikhonov_matrices=None,
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
        self.tikhonov_matrices = tikhonov_matrices

    @property
    def reconstructed_X(self):
        return self.decomposition.construct_slices()

    @property
    def SSE(self):
        return Parafac2_ALS.SSE.fget(self)

    @property
    def MSE(self):
        return Parafac2_ALS.MSE.fget(self)

    @property
    def loss(self):
        loss = self.SSE
        if self.ridge_penalties is not None:
            penalties = self.ridge_penalties
            loss += penalties[0]*np.linalg.norm(self.decomposition.A)**2

            loss += penalties[2]*np.linalg.norm(self.decomposition.B)**2
        if self.tikhonov_matrices is not None:
            for i, tikhonov_matrix in enumerate(self.tikhonov_matrices):
                if tikhonov_matrix is None:
                    continue
                elif i == 0:
                    loss += quadratic_form_trace(tikhonov_matrix, self.decomposition.A)
                elif i == 1:
                    for matrix in self.decomposition.B:
                        loss += quadratic_form_trace(tikhonov_matrix, matrix)
                elif i == 2:
                    loss += quadratic_form_trace(tikhonov_matrix, self.decomposition.C)
        return loss
   
    def _check_valid_components(self, decomposition):
        return Parafac2_ALS._check_valid_components(self, decomposition)

    def set_target(self, X):
        Parafac2_ALS.set_target(self, X)
        self.unfolded_X = np.concatenate([X_slice for X_slice in self.X], axis=1)

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
        super()._init_fit(X, max_its=max_its, initial_decomposition=initial_decomposition)

    def init_svd(self):
        raise NotImplementedError

    def init_random(self):
        self.decomposition = self.DecompositionType.random_init(self.X_shape, rank=self.rank)
    
    def init_cp(self):
        X = np.asarray(self.X)
        cp_als = CP_ALS(self.rank, 20, non_negativity_constraints=self.non_negativity_constraints)
        cp_als.fit(X.transpose(1, 2, 0))
        self.decomposition = self.DecompositionType.from_kruskaltensor(cp_als.decomposition)
        self.pf2_decomposition = Parafac2Tensor.from_kruskaltensor(cp_als.decomposition)

    def init_components(self, initial_decomposition=None):
        if self.init.lower() == 'svd':
            self.init_svd()
        elif self.init.lower() == 'random':
            self.init_random()
        elif self.init.lower() == 'cp':
            self.init_cp()
        elif self.init.lower() == 'from_checkpoint':
            self.load_checkpoint(initial_decomposition)
        elif self.init.lower() == 'precomputed':
            self._check_valid_components(initial_decomposition)
            self.decomposition = initial_decomposition
        elif Path(self.init).is_file():
            self.load_checkpoint(self.init)
        else:
            # TODO: better message
            raise ValueError('Init method must be either `random`, `cp`, `parafac2`, `svd`, `from_checkpoint`, `precomputed` or a path to a checkpoint.')


class CoupledMatrices_ALS(BaseCoupledMatrices):
    parafac2_init_max_its = 50

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
        tikhonov_matrices=None,
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
            print_frequency=print_frequency,
            tikhonov_matrices=tikhonov_matrices,
        )
        self.non_negativity_constraints = non_negativity_constraints
        self.orthonormality_constraints = orthonormality_constraints

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
        if self.non_negativity_constraints is None:
            self.non_negativity_constraints = [False]*3
        if self.orthonormality_constraints is None:
            self.orthonormality_constraints = [False]*3

        super()._init_fit(X, max_its=max_its, initial_decomposition=initial_decomposition)
        for mode, (orthogonality) in enumerate(self.orthonormality_constraints):
            fm = self.decomposition.factor_matrices[mode]
            if orthogonality:
                self.decomposition.factor_matrices[mode] = np.linalg.qr(fm)[0]

        self._rel_function_change = np.inf
        self.prev_loss = self.loss

    def init_parafac2(self):
        X = list(self.X)
        non_negativity_constraints = copy(self.non_negativity_constraints)
        non_negativity_constraints[1] = False
        parafac2_als = Parafac2_ALS(
            self.rank, 
            self.parafac2_init_max_its,
            non_negativity_constraints=non_negativity_constraints,
            init='random',
            print_frequency=-1
        )
        parafac2_als.fit(X)
        A = parafac2_als.decomposition.A
        B = [B_k for B_k in parafac2_als.decomposition.B]
        C = parafac2_als.decomposition.C

        if self.non_negativity_constraints[1]:
            rightsolve = base.non_negative_rightsolve
        else:
            rightsolve = base.rightsolve

        for i in range(self.rank):
            if np.allclose(A[:, i], 0):
                A[:, i] = np.random.standard_uniform(A[:, i].shape)
                
            if np.allclose(C[:, i], 0):
                C[:, i] = np.random.standard_uniform(C[:, i].shape)


        for k, (X_k, D_k) in enumerate(zip(X, C)):
            B[k] = rightsolve((D_k*A).T, X_k.T)
        
        self.pf2_decomposition = parafac2_als.decomposition

        # If B mode is non-negative and one of the other modes can be flipped
        # Let us assume that A is not constrained and B is
        # Then, the PARAFAC2 init (without sign constraints on A and C) might yield
        # a decomposition where a column in A is flipped and a "slice-column" in B
        # is flipped. This means that an originally all-positive B would become all-negative
        # When we run the B update step above, these all-negative B components will be
        # set to zero. Therefore, we test if any B components are zero everywhere, and if so
        # we flip the corresponding A mode and find new B components.
        if not all(self.non_negativity_constraints) and self.non_negativity_constraints[1]:
            all_zero = [True]*A.shape[1]
            for B_k in B:
                for r, B_kr in enumerate(B_k.T):
                    if not np.allclose(B_kr, 0):
                        all_zero[r] = False

            # This is only a problem if B mode is non-negative
            for r, zero_column in enumerate(all_zero):
                if not zero_column:
                    continue
                
                if not self.non_negativity_constraints[0]:
                    A[:, r] *= -1
                elif not self.non_negativity_constraints[2]:
                    C[:, r] *= -1
                else:
                    raise RuntimeError('This should never happen')

            # Compute new B-values
            if any(all_zero):
                for k, (X_k, D_k) in enumerate(zip(X, C)):
                    B[k] = rightsolve((D_k*A).T, X_k.T)

        self.decomposition = self.DecompositionType(A, B, C)

    def init_components(self, initial_decomposition=None):
        if self.init.lower() == 'svd':
            self.init_svd()
        elif self.init.lower() == 'random':
            self.init_random()
        elif self.init.lower() == 'cp':
            self.init_cp()
        elif self.init.lower() == 'parafac2':
            self.init_parafac2()
        elif self.init.lower() == 'from_checkpoint':
            self.load_checkpoint(initial_decomposition)
        elif self.init.lower() == 'precomputed':
            self._check_valid_components(initial_decomposition)
            self.decomposition = initial_decomposition
        elif Path(self.init).is_file():
            self.load_checkpoint(self.init)
        else:
            # TODO: better message
            raise ValueError('Init method must be either `random`, `cp`, `parafac2`, `svd`, `from_checkpoint`, `precomputed` or a path to a checkpoint.')

    def _get_rightsolve(self, mode):
        return CP_ALS._get_rightsolve(self, mode)

    def _update_constant_mode(self):
        rightsolve = self._get_rightsolve(0)
        right = np.concatenate([c*B for c, B in zip(self.decomposition.C, self.decomposition.B)], axis=0)
        self.decomposition.A[...] = rightsolve(right.T, self.unfolded_X) # Hva skal den være?
        self.decomposition.A[...] /= np.linalg.norm(self.decomposition.A, axis=0)

    def _update_evolving_mode(self):
        rightsolve = self._get_rightsolve(1)
        for k, (c_row, factor_matrix) in enumerate(zip(self.decomposition.C, self.decomposition.B)):
            left = (c_row*self.decomposition.A)
            factor_matrix[...] = rightsolve(left.T, self.X[k].T)
            factor_matrix[...] /= np.linalg.norm(factor_matrix, axis=0)


    def _update_evolve_over_mode(self):
        rightsolve = self._get_rightsolve(2)
        for k, (c_row, factor_matrix) in enumerate(zip(self.decomposition.C,  self.decomposition.B)):
            X_k_vec = self.X[k].reshape(-1, 1)
            lhs = base.khatri_rao(self.decomposition.A, factor_matrix)
            c_row[...] = rightsolve(lhs.T, X_k_vec.T).ravel()

    def _update_convergence(self):
        return Parafac2_ALS._update_convergence(self)

    def _update_coupled_matrices_factor_matrices(self):
        self._update_constant_mode()
        self._update_evolving_mode()
        self._update_evolve_over_mode()

    def _fit(self):
        for it in range(self.max_its - self.current_iteration):
            if abs(self._rel_function_change) < self.convergence_tol:
                break

            self._update_coupled_matrices_factor_matrices()
            self._update_convergence()

            if self.current_iteration % self.print_frequency == 0 and self.print_frequency > 0:
                print(f'{self.current_iteration:6d}: The MSE is {self.MSE:4g}, f is {self.loss:4g}, '
                    f'improvement is {self._rel_function_change:g}')

            self._after_fit_iteration()

        # Should this be in self._after_final_iteration()?
        if (
            ((self.current_iteration+1) % self.checkpoint_frequency != 0) and 
            (self.checkpoint_frequency > 0)
        ):
            self.store_checkpoint()


class FlexibleParafac2_ALS(CoupledMatrices_ALS):
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
        signal_to_noise=-1,
        tikhonov_matrices=None,
        coupling_strength=None,
        normalise_tensor=True
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
            print_frequency=print_frequency,
            non_negativity_constraints=non_negativity_constraints,
            orthonormality_constraints=orthonormality_constraints,
            tikhonov_matrices=tikhonov_matrices,
        )
        self.signal_to_noise = signal_to_noise
        self._coupling_strength = coupling_strength
        self.normalise_tensor = normalise_tensor

    def _get_rightsolve(self, mode, k=None):
        rightsolve = CP_ALS._get_rightsolve(self, mode)
        if mode == 1:
            if k is None:
                raise ValueError('Must supply value for ``k`` if ``mode == 1``.')
            
            rightsolve = base.add_rightsolve_coupling(
                rightsolve,
                self.pf2_decomposition.B[k],
                self.coupling_penalties[k]
            )

        return rightsolve

    def _update_evolving_mode(self):
        for k, (c_row, factor_matrix) in enumerate(zip(self.decomposition.C, self.decomposition.B)):
            rightsolve = self._get_rightsolve(1, k=k)
            left = (c_row*self.decomposition.A)
            factor_matrix[...] = rightsolve(left.T, self.X[k].T)

    def _update_parafac2_factor_matrices(self):
        # update projection matrices
        for k, projection_matrix in enumerate(self.pf2_decomposition.projection_matrices):
            projection_matrix[...] = base.orthogonal_rightsolve(
                self.pf2_decomposition.blueprint_B,
                self.decomposition.B[k]
            )

        # estimate blueprint B
        self.pf2_decomposition.blueprint_B[...] *= 0
        for k, projection_matrix in enumerate(self.pf2_decomposition.projection_matrices):
            self.pf2_decomposition.blueprint_B[...] += self.coupling_penalties[k]*projection_matrix.T@self.decomposition.B[k]
        
        self.pf2_decomposition.blueprint_B[...] /= self.coupling_penalties.sum()
        self.pf2_decomposition.blueprint_B[...] /= np.linalg.norm(self.pf2_decomposition.blueprint_B, axis=0)
        
    def _fit(self):
        for it in range(self.max_its - self.current_iteration):
            if abs(self._rel_function_change) < self.convergence_tol:
                break

            self._update_parafac2_factor_matrices()
            self._update_coupled_matrices_factor_matrices()
            self._update_coupling_penalty(it)
            self._update_convergence()

            if self.current_iteration % self.print_frequency == 0 and self.print_frequency > 0:
                print(f'{self.current_iteration:6d}: The MSE is {self.MSE:4g}, f is {self.loss:4g}, '
                      f'improvement is {self._rel_function_change:g}')
                print(f'        The coupling error is {self.coupling_error}')

            self._after_fit_iteration()

        # Should this be in self._after_final_iteration()?
        if (
            ((self.current_iteration+1) % self.checkpoint_frequency != 0) and 
            (self.checkpoint_frequency > 0)
        ):
            self.store_checkpoint()

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
        self.pf2_decomposition = None
        if self.normalise_tensor:
            X_norm = np.linalg.norm(np.array(X))
            for X_k in X:
                pass
                X_k[...] /= X_norm
        
        super()._init_fit(X, max_its=max_its, initial_decomposition=initial_decomposition)
        
        if self.non_negativity_constraints is None:
            self.non_negativity_constraints = [False]*3
        if self.orthonormality_constraints is None:
            self.orthonormality_constraints = [False]*3
        for mode, (orthogonality) in enumerate(self.orthonormality_constraints):
            fm = self.decomposition.factor_matrices[mode]
            if orthogonality:
                self.decomposition.factor_matrices[mode] = np.linalg.qr(fm)[0]
        
        self._rel_function_change = np.inf
        self.prev_loss = self.loss

        self.coupling_penalties = self._init_coupled_penalties()

        if self.pf2_decomposition is None:
            B = np.random.rand(self.rank, self.rank)
            P = [np.eye(*B_k.shape) for B_k in self.decomposition.B]
            self.pf2_decomposition = decompositions.Parafac2Tensor(A=self.decomposition.A, blueprint_B=B, projection_matrices=P, C=self.decomposition.C) 

    def _init_coupled_penalties(self):
        self.coupled_penalties = np.empty((self.decomposition.shape[-1]))
        estimated_X = self.decomposition.construct_slices()

        for k, (X_slice, estimated_X_slice) in enumerate(zip(self.X, estimated_X)):
            evolving_factor = self.decomposition.B[k]
            SSE = np.linalg.norm(X_slice - estimated_X_slice)**2
            self.coupled_penalties[k] = SSE/(np.linalg.norm(evolving_factor))**2
        
        return self.coupled_penalties

    def _update_coupling_penalty(self, it):
        if self._coupling_strength is not None:
            self.coupling_penalties[:] = self._coupling_strength
            return
        if it == 0:
            scale = 10**(-self.signal_to_noise/10)
            estimated_X = self.decomposition.construct_slices()

            for k, (X_slice, estimated_X_slice) in enumerate(zip(self.X, estimated_X)):
                SSE = np.linalg.norm(X_slice - estimated_X_slice)**2
                coupling_error = np.linalg.norm(self.decomposition.B[k] - self.pf2_decomposition.B[k])**2

                self.coupling_penalties[k] = scale*SSE/coupling_error

        else:
            for k, coupling_penalty in enumerate(self.coupling_penalties):
                self.coupling_penalties[k] = min(10, coupling_penalty*1.02)

    
    @property
    def parafac2_error(self):
        return utils.slice_SSE(self.X, self.pf2_decomposition.construct_slices())

    @property
    def coupling_error(self):
        ce = 0
        num_timesteps = len(self.decomposition.B)

        for k, B in enumerate(self.decomposition.B):
            ce += np.linalg.norm(B - self.pf2_decomposition.B[k])**2

        return ce/(self.rank*num_timesteps)

    def store_checkpoint(self):
        super().store_checkpoint()
        with h5py.File(self.checkpoint_path, 'a') as h5:
            checkpoint_group = h5.create_group(f'parafac2_checkpoint_{self.current_iteration:05d}')
            self.pf2_decomposition.store_in_hdf5_group(checkpoint_group)

    def load_checkpoint(self, checkpoint_path, load_it=None):
        """Load the specified checkpoint at the given iteration.

        If ``load_it=None``, then the latest checkpoint will be used.
        """
        # TODO: classmethod, dump all params. Requires major refactoring.
        super().load_checkpoint(checkpoint_path, load_it=load_it)

        with h5py.File(checkpoint_path) as h5:
            group_name = f'parafac2_checkpoint_{self.current_iteration:05d}'
            if group_name not in h5:
                raise ValueError(
                    f'There is no PARAFAC2 checkpoint {group_name} in {checkpoint_path}\n'
                    'This should never happen since that means that the evolving tensor was stored,\n'
                    'but not the PARAFAC2 decomposition.'
                    )

            checkpoint_group = h5[group_name]
            pf2_decomposition = decompositions.Parafac2Tensor.load_from_hdf5_group(checkpoint_group)

        self._check_valid_components(pf2_decomposition)
        self.pf2_decomposition = pf2_decomposition
