from copy import copy
from pathlib import Path
import h5py
    
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla
import scipy.sparse as sparse
from sklearn.linear_model import Lasso

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from .. import base
from .cp import get_sse_lhs
from .parafac2 import BaseParafac2, compute_projected_X, Parafac2_ALS
from .decompositions import KruskalTensor, Parafac2Tensor
from .base_decomposer import BaseDecomposer
from . import decompositions
from .. import utils
from .utils import quadratic_form_trace
from ._tv_prox import TotalVariation


# Default callback
def noop(*args, **kwargs):
    pass


class BaseSubProblem:
    def __init__(self):
        pass

    def update_decomposition(self, X, decomposition):
        pass

    def regulariser(self, factor) -> float:
        return 0


class NotUpdating(BaseSubProblem):
    def update_decomposition(self, X, decomposition):
        pass


class RLS(BaseSubProblem):
    def __init__(self, mode, ridge_penalty=0, non_negativity=False):
        self.ridge_penalty = ridge_penalty
        self.non_negativity = non_negativity
        self.mode = mode
        self._matrix_khatri_rao_product_cache = None
    
    def update_decomposition(self, X, decomposition):
        lhs = get_sse_lhs(decomposition.factor_matrices, self.mode)
        rhs = base.matrix_khatri_rao_product(X, decomposition.factor_matrices, self.mode)

        self._matrix_khatri_rao_product_cache = rhs

        rightsolve = self._get_rightsolve()

        decomposition.factor_matrices[self.mode][:] = rightsolve(lhs, rhs)
    
    def _get_rightsolve(self):
        rightsolve = base.rightsolve
        if self.non_negativity:
            rightsolve = base.non_negative_rightsolve
        if self.ridge_penalty:
            rightsolve = base.add_rightsolve_ridge(rightsolve, self.ridge_penalty)
        return rightsolve


def prox_reg_lstsq(A, B, reg, C, D):
    """Solve ||AX - B||^2 + r||CX - D||^2
    """
    # We can save much time here by storing a QR decomposition of A_new.
    reg = np.sqrt(reg/2)
    A_new = np.concatenate([A, reg*C], axis=0)
    B_new = np.concatenate([B, reg*D], axis=0)
    return np.linalg.lstsq(A_new, B_new)[0]


class ADMMSubproblem(BaseSubProblem):
    def __init__(self, mode, rho, tol=1e-3, max_it=50, non_negativity=True):
        self.non_negativity = non_negativity
        self.rho = rho
        self.mode = mode
        self.tol = tol
        self.max_it = max_it
        
        self._callback = noop

    def callback(self, X, decomposition, fm, aux_fm, dual_variable):
        """Calls self._callback, which should have the following signature:
        self._callback(self, X, decomposition, fm, aux_fm, dual_variable)

        By default callback does nothing.
        """
        self._callback(self, X, decomposition, fm, aux_fm, dual_variable)
    
    def update_decomposition(self, X, decomposition):
        # TODO: Cache QR decomposition of lhs.T
        # ||MA - X||
        # M = lhs
        # X = rhs
        lhs = base.khatri_rao(
            *decomposition.factor_matrices, skip=self.mode
        ).T
        rhs = base.unfold(X, self.mode)

        # Assign name to current factor matrix to and identity reduce space
        fm = decomposition.factor_matrices[self.mode]
        I = np.identity(fm.shape[1])
        
        # Initialise main variable by unregularised least squares,
        # auxiliary variable by projecting main variable init
        # and the dual variable as zeros
        fm[:] = np.linalg.lstsq(lhs.T, rhs.T)[0].T
        aux_fm = self.init_aux_factor_matrix(decomposition)
        dual_variable = np.zeros_like(fm)

        # Update decomposition and auxiliary variable with proximal
        # map calls followed by a gradient ascent step for the dual
        # variable. Stop if main and aux variable are close enough
        for _ in range(self.max_it):
            if self.has_converged(fm, aux_fm):
                break
            fm[:] = prox_reg_lstsq(lhs.T, rhs.T, self.rho/2, I, aux_fm.T - dual_variable.T).T
            self.update_constraint(decomposition, aux_fm, dual_variable)

            dual_variable += fm - aux_fm
            self.callback(self, X, decomposition, fm, aux_fm)
        
        # Use aux variable for hard constraints
        decomposition.factor_matrices[self.mode][:] = aux_fm
    
    def init_aux_factor_matrix(self, decomposition):
        """Initialise the auxiliary factor matrix used to fit the constraints.
        """
        return np.maximum(decomposition.factor_matrices[self.mode], 0)
    
    def update_constraint(self, decomposition, aux_fm, dual_variable):
        """Update the auxiliary factor matrix used to fit the constraints inplace.
        """
        np.maximum(
            decomposition.factor_matrices[self.mode] + dual_variable,
            0,
            out=aux_fm
        )
    
    def has_converged(self, fm, aux_fm):
        return np.linalg.norm(fm - aux_fm) < self.tol


class BaseParafac2SubProblem(BaseSubProblem):
    _is_pf2_evolving_mode = True
    mode = 1
    def __init__(self):
        pass

    def minimise(self, X, decomposition, projected_X=None, should_update_projections=True):
        pass


class Parafac2RLS(BaseParafac2SubProblem):
    SKIP_CACHE = False
    def __init__(self, ridge_penalty=0):
        self.ridge_penalty = ridge_penalty
        self.non_negativity = False
        self._callback = noop

    def compute_projected_X(self, projection_matrices, X, out=None):
        return compute_projected_X(projection_matrices, X, out=out)

    def update_projections(self, X, decomposition):
        K = len(X)

        for k in range(K):
            A = decomposition.A
            C = decomposition.C
            blueprint_B = decomposition.blueprint_B

            decomposition.projection_matrices[k][:] = base.orthogonal_solve(
                (C[k]*A)@blueprint_B.T,
                X[k]
            ).T

    def update_decomposition(
        self, X, decomposition, projected_X=None, should_update_projections=True
    ):
        """Updates the decomposition inplace

        If a projected data tensor is supplied, then it is updated inplace
        """
        if should_update_projections:
            self.update_projections(X, decomposition)
            projected_X = self.compute_projected_X(decomposition.projection_matrices, X, out=projected_X)
        
        if projected_X is None:
            projected_X = self.compute_projected_X(decomposition.projection_matrices, X, out=projected_X)
        ktensor = KruskalTensor([decomposition.A, decomposition.blueprint_B, decomposition.C])
        RLS.update_decomposition(self, X=projected_X, decomposition=ktensor)

    def _get_rightsolve(self):
        return RLS._get_rightsolve(self)


def safe_factorise(array):
    if sparse.issparse(array):
        return spla.splu(array)
    else:
        return sla.cho_factor(array)


def safe_factor_solve(factor, x):
    if isinstance(factor, spla.SuperLU):
        return factor.solve(x)
    else:
        return sla.cho_solve(factor, x)


def evolving_factor_total_variation(factor):
    return TotalVariation(factor, 1).center_penalty()


def total_variation_prox(factor, strength):
    return TotalVariation(factor, strength).prox()


class Parafac2ADMM(BaseParafac2SubProblem):
    """
    To add new regularisers:
        * Change __init__
        * Change constraint_prox
        * Change regulariser
    """
    # In our notes: U -> dual variable
    #               \tilde{B} -> aux_fms
    #               B -> decomposition
    SKIP_CACHE = False
    def __init__(
        self,
        rho=None,
        tol=1e-3,
        max_it=50,
        non_negativity=False,
        l2_similarity=None,
        l1_penalty=None,
        tv_penalty=None,
        temporal_similarity=0,
        verbose=False,
        decay_num_it=False,
        num_it_converge_at=30,
        num_it_converge_to=5,
        cache_components=True,
    ):
        if rho is None:
            self.auto_rho = True
        else:
            self.auto_rho = False

        self.rho = rho
        self.tol = tol
        self._max_it = max_it
        
        self._decay_num_it = decay_num_it
        self._num_it_converge_to = num_it_converge_to
        self._num_it_decay_rate = (num_it_converge_to/max_it)**(1/num_it_converge_at)

        self.non_negativity = non_negativity
        self.l2_similarity = l2_similarity
        self.l1_penalty = l1_penalty
        self.tv_penalty = tv_penalty
        self.temporal_similarity = temporal_similarity

        if self.temporal_similarity > 0:
            raise RuntimeError("This doesn't work")

        if self.temporal_similarity > 0 and l2_similarity is None:
            self.l2_similarity = 0
        if non_negativity and l2_similarity is not None:
            raise ValueError("Not implemented non negative similarity")
        if l2_similarity is not None and l1_penalty:
            raise ValueError("Not implemented L1+L2 with similarity")

        self.verbose = verbose
        self._qr_cache = None
        self._reg_factor_cache = None
        self.dual_variables = None
        self.aux_fms = None
        self.it_num = 0

        self._cache_components = cache_components
        
        self._callback = noop

    def callback(self, X, decomposition, aux_fms, dual_variable, init=False):
        """Calls self._callback, which should have the following signature:
        self._callback(self, X, decomposition, fm, aux_fm, dual_variable)

        By default callback does nothing.
        """
        self._callback(self, X, decomposition, aux_fms, dual_variable, init)
    
    @property
    def max_it(self):
        converge_to = self._num_it_converge_to
        if not self._decay_num_it or self._max_it <= converge_to:
            return self._max_it
        
        self._max_it = int(self._num_it_decay_rate*self._max_it)
        return self._max_it

    
    def update_decomposition(
        self, X, decomposition, projected_X=None, should_update_projections=True
    ):
        # Clear caches
        self._qr_cache = None
        self._reg_factor_cache = None

        # Compute rho
        if self.auto_rho:
            self.rho, self._qr_cache = self.compute_auto_rho(decomposition)

        # Init constraint by projecting the decomposition
        if self.aux_fms is None or self.it_num == 1 or (not self._cache_components):
            aux_fms = self.init_constraint(decomposition)
        else:
            aux_fms = self.aux_fms

        # Init dual variables
        if self.dual_variables is None or self.it_num == 1 or (not self._cache_components):
            dual_variables = [np.zeros_like(aux_fm) for aux_fm in aux_fms]
        else:
            dual_variables = self.dual_variables

        if projected_X is None:
            projected_X = self.compute_projected_X(decomposition.projection_matrices, X, out=projected_X)

        # The decomposition is modified inplace each iteration
        for i in range(self.max_it):
            self.callback(X, decomposition, aux_fms, dual_variables, init=(i==0))
            self.update_blueprint(X, decomposition, aux_fms, dual_variables, projected_X)

            if should_update_projections:
                self.update_projections(X, decomposition, aux_fms, dual_variables)
                projected_X = self.compute_projected_X(decomposition.projection_matrices, X, out=projected_X)

            self.update_blueprint(X, decomposition, aux_fms, dual_variables, projected_X)
            old_aux_fms = aux_fms
            aux_fms = self.compute_next_aux_fms(decomposition, dual_variables)
            self.update_dual(decomposition, aux_fms, dual_variables)

            if self.has_converged(decomposition, aux_fms, old_aux_fms, dual_variables):
                break

        self.callback(X, decomposition, aux_fms, dual_variables, init=False)
        self.dual_variables = dual_variables
        self.aux_fms = aux_fms
        self.it_num += 1

    def compute_auto_rho(self, decomposition):     
        lhs = base.khatri_rao(
            decomposition.A, decomposition.C,
        )
        rho = np.linalg.norm(lhs)**2/decomposition.rank

        reg_lhs = np.vstack([np.identity(decomposition.rank) for _ in decomposition.B])
        reg_lhs *= np.sqrt(rho/2)
        lhs = np.vstack([lhs, reg_lhs])

        return rho, np.linalg.qr(lhs)

    def init_constraint(self, decomposition):
        init_P, init_B = decomposition.projection_matrices, decomposition.blueprint_B
        B = [P_k@init_B for P_k in init_P]
        return [
            self.constraint_prox(B_k, decomposition) for k, B_k in enumerate(B)
        ]

    def compute_next_aux_fms(self, decomposition, dual_variables):
        projections = decomposition.projection_matrices
        blueprint_B = decomposition.blueprint_B
        rank = blueprint_B.shape[1]
        
        Bks = [P_k@blueprint_B for P_k in projections]
        Bks = np.concatenate(Bks, axis=1)
        dual_variables = np.concatenate([dual_variable for dual_variable in dual_variables], axis=1)

        proxed = self.constraint_prox(Bks + dual_variables, decomposition)
        return [
            proxed[:, k*rank:(k+1)*rank] for k, _ in enumerate(projections)
        ]


        return [
            self.constraint_prox(
                P_k@blueprint_B + dual_variables[k], decomposition,
            ) for k, P_k in enumerate(projections)
        ]
    
    def constraint_prox(self, x, decomposition):
        if self.non_negativity and self.l1_penalty:
            return np.maximum(x - 2*self.l1_penalty/self.rho, 0)
        elif self.tv_penalty:
            return total_variation_prox(x, 2*self.tv_penalty/self.rho)
        elif self.non_negativity:
            return np.maximum(x, 0)      
        elif self.l1_penalty:
            return np.sign(x)*np.maximum(np.abs(x) - 2*self.l1_penalty/self.rho, 0)
        elif self.l2_similarity is not None:
            similar_to=0
            step_length=0
            # if k == 0:
            #     step_length = 1
            #     similar_to = decomposition.B[k+1]
            # elif k == (decomposition.shape[2] - 1):
            #     step_length = 1
            #     similar_to = decomposition.B[k-1]
            # else:
            #     step_length = 2
            #     similar_to = decomposition.B[k-1] + decomposition.B[k+1]


            if self.SKIP_CACHE:
                I = np.identity(self.l2_similarity.shape[0])
                reg_matrix = self.l2_similarity + (0.5*self.rho + self.temporal_similarity*step_length)*I
                return np.linalg.solve(reg_matrix, 0.5*self.rho*x + self.temporal_similarity*similar_to)

            if self._reg_factor_cache is None:
                I = np.identity(self.l2_similarity.shape[0])
                reg_matrix = self.l2_similarity + (0.5*self.rho + self.temporal_similarity*step_length)*I

                factor = safe_factorise(sparse.csc_matrix(reg_matrix))
                self._reg_factor_cache = factor
            else:
                factor = self._reg_factor_cache
            
            rhs = 0.5*self.rho*x + self.temporal_similarity*similar_to
            return safe_factor_solve(factor, rhs)
        else:
            return x

    def update_dual(self, decomposition, aux_fms, dual_variables):
        for P_k, aux_fm, dual_variable in zip(decomposition.projection_matrices, aux_fms, dual_variables):
            B_k = P_k@decomposition.blueprint_B
            dual_variable += B_k - aux_fm  # TODO: Look at this one a bit more

    def update_projections(self, X, decomposition, aux_fms, dual_variable):
        # Triangle equation from notes
        A = decomposition.A
        blueprint_B = decomposition.blueprint_B
        C = decomposition.C
        # TODO: randomise order
        for k, X_k in enumerate(X):
            unreg_lhs = (A*C[k])@(blueprint_B.T)
            reg_lhs = np.sqrt(self.rho/2)*(blueprint_B.T)
            lhs = np.vstack((unreg_lhs, reg_lhs))

            unreg_rhs = X_k
            reg_rhs = np.sqrt(self.rho/2)*(aux_fms[k] - dual_variable[k]).T
            rhs = np.vstack((unreg_rhs, reg_rhs))
            
            decomposition.projection_matrices[k][:] = base.orthogonal_solve(lhs, rhs).T

    def update_blueprint(self, X, decomposition, aux_fms, dual_variables, projected_X):
        # Square equation from notes
        if self._qr_cache is None:
            lhs = base.khatri_rao(
                decomposition.A, decomposition.C,
            )
            reg_lhs = np.vstack([np.identity(decomposition.rank) for _ in aux_fms])
            reg_lhs *= np.sqrt(self.rho/2)
            lhs = np.vstack([lhs, reg_lhs])
            self._qr_cache = np.linalg.qr(lhs)
        Q, R = self._qr_cache
        
        rhs = base.unfold(projected_X, 1).T
        projected_aux = [
            (aux_fm - dual_variable).T@projection
            for aux_fm, dual_variable, projection in zip(
                aux_fms, dual_variables, decomposition.projection_matrices
            )
        ]
        reg_rhs = np.vstack(projected_aux)
        reg_rhs *= np.sqrt(self.rho/2)
        rhs = np.vstack([rhs, reg_rhs])

        decomposition.blueprint_B[:] = np.linalg.solve(R, Q.T@rhs).T
        #decomposition.blueprint_B[:] = prox_reg_lstsq(lhs, rhs, self.rho, reg_lhs, reg_rhs).T
    
    def compute_projected_X(self, projection_matrices, X, out=None):
        return compute_projected_X(projection_matrices, X, out=out)

    def _compute_relative_duality_gap(self, fms, aux_fms):
        gap_sq = sum(np.linalg.norm(fm - aux_fm)**2 for fm, aux_fm in zip(fms, aux_fms))
        aux_norm_sq = sum(np.linalg.norm(aux_fm)**2 for aux_fm in aux_fms)
        return gap_sq/aux_norm_sq

    def has_converged(self, decomposition, aux_fms, old_aux_fms, dual_variables):
        duality_gap = self._compute_relative_duality_gap(decomposition.B, aux_fms)
        aux_change_sq = sum(
            np.linalg.norm(aux_fm - old_aux_fm)**2 for aux_fm, old_aux_fm in zip(aux_fms, old_aux_fms)
        )
        dual_var_norm_sq = sum(np.linalg.norm(dual_var)**2 for dual_var in dual_variables)
        aux_change_criterion = (aux_change_sq + 1e-16) / (dual_var_norm_sq + 1e-16)
        if self.verbose:
            print("primal criteria", duality_gap, "dual criteria", aux_change_sq)

        
        return duality_gap < self.tol and aux_change_criterion < self.tol
    
    def regulariser(self, factor_matrices):
        reg = 0
        if self.l2_similarity is not None:
            factor_matrices = np.array(factor_matrices)
            reg += sum(
                quadratic_form_trace(self.l2_similarity, factor_matrix)
                for factor_matrix in factor_matrices
            )
        if self.l1_penalty is not None:
            factor_matrices = np.array(factor_matrices)
            reg += self.l1_penalty*np.linalg.norm(factor_matrices.ravel(), 1)
        if self.tv_penalty is not None:
            factor_matrices = np.array(factor_matrices)
            reg += self.tv_penalty*evolving_factor_total_variation(factor_matrices)
        if self.temporal_similarity > 0:
            for k, B_k in enumerate(factor_matrices):   # <- This should not be B_k
                if k > 0:
                    reg += self.temporal_similarity*(np.linalg.norm(B_k - factor_matrices[k-1])**2)
                if k < len(factor_matrices) - 1:
                    reg += self.temporal_similarity*(np.linalg.norm(B_k - factor_matrices[k+1])**2)
        return reg
        

class FlexibleParafac2ADMM(BaseParafac2SubProblem):
    def __init__(self, non_negativity=True):
        self.non_negativity = non_negativity

    def update_decomposition(self,  X, decomposition, projected_X, update_projections):
        pass

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
        convergence_check_frequency=1,
    ):
        if (
            not hasattr(sub_problems[1], '_is_pf2_evolving_mode') or 
            not sub_problems[1]._is_pf2_evolving_mode
        ):
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
        self.convergence_check_frequency = convergence_check_frequency

    def _check_valid_components(self, decomposition):
        return BaseParafac2._check_valid_components(self, decomposition)

    @property
    def regularisation_penalty(self):
        factor_matrices = [
            self.decomposition.A,
            np.array(self.decomposition.B),
            self.decomposition.C
        ]
        return sum(sp.regulariser(fm) for sp, fm in zip(self.sub_problems, factor_matrices))

    @property
    def loss(self):
        factor_matrices = [
            self.decomposition.A,
            np.array(self.decomposition.B),
            self.decomposition.C
        ]
        return (
            self.SSE + 
            sum(sp.regulariser(fm) for sp, fm in zip(self.sub_problems, factor_matrices))
        )

    def _update_parafac2_factors(self):
        should_update_projections = self.current_iteration % self.projection_update_frequency == 0
        # The function below updates the decomposition and the projected X inplace.
        # print(f'Before {self.current_iteration:6d}A: The MSE is {self.MSE:4g}, f is {self.loss:4g}, '
        #               f'improvement is {self._rel_function_change:g}')
        decomposition = self.decomposition
        self.sub_problems[1].update_decomposition(
            self.X, self.decomposition, self.projected_X, should_update_projections=should_update_projections
        )
        # print(f'Before {self.current_iteration:6d}B: The MSE is {self.MSE:4g}, f is {self.loss:4g}, '
        #               f'improvement is {self._rel_function_change:g}')
        self.sub_problems[0].update_decomposition(
            self.projected_X, self.cp_decomposition
        )
        # print(f'Before {self.current_iteration:6d}C: The MSE is {self.MSE:4g}, f is {self.loss:4g}, '
        #               f'improvement is {self._rel_function_change:g}')
        self.sub_problems[2].update_decomposition(
            self.projected_X, self.cp_decomposition
        )

    def _fit(self):
        for it in range(self.max_its - self.current_iteration):
            if self._has_converged():
                break

            self._update_parafac2_factors()
            self._after_fit_iteration()

            if self.current_iteration % self.print_frequency == 0 and self.print_frequency > 0:
                rel_change = np.asscalar(np.array(self._rel_function_change))

                print(f'{self.current_iteration:6d}: The MSE is {self.MSE:4g}, f is {np.asscalar(self.loss):4g}, '
                      f'improvement is {rel_change:g}')

        if (
            ((self.current_iteration+1) % self.checkpoint_frequency != 0) and 
            (self.checkpoint_frequency > 0)
        ):
            self.store_checkpoint()

    def init_components(self, initial_decomposition=None):
        if self.init == 'ALS':
            self.pf2 = Parafac2_ALS(self.rank, max_its=100, print_frequency=-1)
            self.pf2.fit([Xi for Xi in self.X])
            self.decomposition = self.pf2.decomposition
        else:
            BaseParafac2.init_components(self, initial_decomposition=initial_decomposition)

    def _has_converged(self):
        has_converged = False
        if self.current_iteration % self.convergence_check_frequency == 0 and self.current_iteration > 0:
            loss = self.loss
            tol = (1 - self.convergence_tol)**self.convergence_check_frequency
            has_converged = loss >= tol*self.prev_loss

            self._rel_function_change = (self.prev_loss - loss)/self.prev_loss
            self.prev_loss = loss
        return has_converged

    def _init_fit(self, X, max_its, initial_decomposition):
        super()._init_fit(X=X, max_its=max_its, initial_decomposition=initial_decomposition)
        self.cp_decomposition = KruskalTensor(
            [self.decomposition.A, self.decomposition.blueprint_B, self.decomposition.C]
        )
        self.projected_X = compute_projected_X(self.decomposition.projection_matrices, self.X)
        self.prev_loss = self.loss
        self._rel_function_change = np.inf
    
    def init_random(self):
        return BaseParafac2.init_random(self)
    
    def init_svd(self):
        return BaseParafac2.init_svd(self)

    def init_cp(self):
        return BaseParafac2.init_cp(self)

    @property
    def reconstructed_X(self):
        return self.decomposition.construct_slices()
    
    def set_target(self, X):
        BaseParafac2.set_target(self, X)

    @property
    def SSE(self):
        return utils.slice_SSE(self.X, self.reconstructed_X)

    @property
    def MSE(self):
        return self.SSE/self.decomposition.num_elements