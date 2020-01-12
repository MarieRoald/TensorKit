from copy import copy
from pathlib import Path

import h5py
import numpy as np

from .. import base
from .cp import get_sse_lhs
from .parafac2 import BaseParafac2, compute_projected_X
from .decompositions import KruskalTensor, Parafac2Tensor
from .base_decomposer import BaseDecomposer
from . import decompositions


class BaseSubProblem:
    def __init__(self):
        pass

    def minimise(self, X, decomposition) -> np.ndarray:
        pass

    def regulariser(self, factor) -> float:
        pass


class RLS(BaseSubProblem):
    def __init__(self, mode, ridge_penalty=0):
        self.ridge_penalty = ridge_penalty
        self.mode = mode
        self._matrix_khatri_rao_product_cache = None
    
    def minimise(self, X, decomposition):
        lhs = get_sse_lhs(decomposition.factor_matrices, self.mode)
        rhs = base.matrix_khatri_rao_product(X, decomposition.factor_matrices, self.mode)

        self._matrix_khatri_rao_product_cache = rhs

        rightsolve = self._get_rightsolve()
        return rightsolve(lhs, rhs)
    
    def _get_rightsolve(self):
        rightsolve = base.rightsolve
        if self.ridge_penalty:
            rightsolve = base.add_rightsolve_ridge(rightsolve, self.ridge_penalty)
        return rightsolve
    

class BaseParafac2SubProblem(BaseSubProblem):
    _is_pf2_evolving_mode = True
    mode = 1
    def __init__(self):
        pass

    def minimise(self, X, decomposition, projected_X=None, should_update_projections=True):
        pass


class Parafac2RLS(BaseParafac2SubProblem):
    def __init__(self, ridge_penalty=0):
        self.ridge_penalty=ridge_penalty

    def compute_projected_X(self, X, decomposition):
        return compute_projected_X(X, decomposition.projection_matrices)

    def minimise(self, X, decomposition, projected_X=None, should_update_projections=True):
        if should_update_projections:
            projections = compute_projections(X, decomposition.projection_matrices)
            projected_X = self.compute_projected_X(X, projections)
        
        if projected_X is None:
            projected_X = self.compute_projected_X(X, decomposition)
        blueprint_B = RLS.minimise(projected_X, decomposition)
    
    def _get_rightsolve(self):
        return RLS._get_rightsolve(self)
    


class BlockParafac2(BaseDecomposer):
    DecompositionType = decompositions.Parafac2Tensor
    def __init__(
        self,
        rank,
        sub_problems,
        max_its=1000,
        convergence_tol=1e-6,
        init='random',
        loggers=None,
        checkpoint_frequency=None,
        checkpoint_path=None,
        print_frequency=None,
        projection_update_frequency=5,
    ):
        if not hasattr(sub_problems[1], '_is_pf2_evolving_mode') or not sub_problems[1]._is_pf2:
            raise ValueError(
                'Second sub problem must follow PARAFAC2 constraints. If it does, '
                'ensure that `sub_problem._is_pf2 == True`.'
            )

        super().__init__(
            max_its=max_its,
            convergence_tol=convergence_tol,
            loggers=loggers,
            checkpoint_frequency=checkpoint_frequency,
            checkpoint_path=checkpoint_path,
            print_frequency=print_frequency,
        )
        self.rank = rank
        self.sub_problems = sub_problems
        self.init = init
        self.projection_update_frequency = projection_update_frequency

    def _check_valid_components(self, decomposition):
        return BaseParafac2._check_valid_components(self, decomposition)

    def loss(self):
        factor_matrices = [
            self.decomposition.A,
            (self.decomposition.projection_matrices,
            self.decomposition.B), self.decomposition.C
        ]
        return (
            self.SSE + 
            sum(sp.regulariser(fm) for sp, fm in zip(self.sub_problems, factor_matrices))
        )

    def _update_parafac2_factors(self):
        should_update_projections = self.current_iteration % self.projection_update_frequency
        out = self.sub_problems[1].minimise(
            self.X, self.decomposition, should_update_projections=should_update_projections
        )
        next_P = out[0]
        self.decomposition.B[:] = out[1]
        P_update_status = out[2]
        self.projected_X = out[3]

        if P_update_status:
            for P, new_P in zip(self.decomposition.projection_matrices, next_P):
                P[:] = new_P
        
        self.decomposition.A[:] = self.sub_problems[0].minimise(
            self.projected_X, self.cp_decomposition
        )
        self.decomposition.C[:] = self.sub_problems[2].minimise(
            self.projected_X, self.cp_decomposition
        )

    def _fit(self):
        for it in range(self.max_its - self.current_iteration):
            if self._rel_function_change < self.convergence_tol:
                break

            self._update_parafac2_factors()

            if self.current_iteration % self.print_frequency == 0 and self.print_frequency > 0:
                print(f'{self.current_iteration:6d}: The MSE is {self.MSE:4g}, f is {self.loss:4g}, '
                      f'improvement is {self._rel_function_change:g}')

            self._after_fit_iteration()


        if (
            ((self.current_iteration+1) % self.checkpoint_frequency != 0) and 
            (self.checkpoint_frequency > 0)
        ):
            self.store_checkpoint()

    def init_components(self, initial_decomposition=None):
        BaseParafac2.init_components(self, initial_decomposition=initial_decomposition)
        self.cp_decomposition = KruskalTensor(
            (self.decomposition.A, self.decomposition.blueprint_B, self.decomposition.C)
        )
    
    def init_random(self):
        return BaseParafac2.init_random(self)
    
    def init_svd(self):
        return BaseParafac2.init_svd(self)

    def init_cp(self):
        return BaseParafac2.init_cp(self)

    def reconstructed_X(self):
        self.decomposition.construct_slices()
    
    def set_target(self, X):
        BaseParafac2.set_target(self, X)
