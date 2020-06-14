import pytest
from tenkit.decomposition import KruskalTensor, Parafac2Tensor
from tenkit.decomposition import RLS, Parafac2RLS, BlockParafac2, ADMMSubproblem, Parafac2ADMM
from scipy.optimize import approx_fprime
import numpy as np
from tenkit.decomposition.cp import get_sse_lhs
from tenkit.base import matrix_khatri_rao_product
from tenkit import base




class BaseTestSubproblem():
    SubProblem = None

    @pytest.fixture
    def random_rank4_ktensor(self):
        return KruskalTensor.random_init([30, 40, 50], rank=4)
    
    @pytest.fixture
    def random_nonnegative_rank4_ktensor(self):
        return KruskalTensor.random_init([30, 40, 50], 4, random_method='uniform')

    def check_gradient(self, decomposition, **kwargs):
        X = decomposition.construct_tensor()
        wrong_decomposition = KruskalTensor.random_init(
            decomposition.shape,
            rank=decomposition.rank,
            random_method='uniform'
        )
        num_modes = len(decomposition.factor_matrices)

        letters = [chr(ord('i') + mode) for mode in range(num_modes)]
        einsum_pattern = ''.join(f'{l}r, ' for l in letters)[:-2] + ' -> ' + ''.join(letters)

        for mode in range(num_modes):
            print('kwargs', kwargs)
            rls = self.SubProblem(mode=mode, **kwargs)
            rls.update_decomposition(X, wrong_decomposition)

            def loss(x):
                factor_matrices = [
                    fm for fm in wrong_decomposition.factor_matrices
                ]
                factor_matrices[mode] = x.reshape(factor_matrices[mode].shape)
                
                estimated = np.einsum(einsum_pattern, *factor_matrices)
                return np.linalg.norm(estimated - X)**2
            
            raveled_fm = wrong_decomposition.factor_matrices[mode].ravel()
            deriv = approx_fprime(
                raveled_fm,
                loss,
                epsilon=np.sqrt(np.finfo(float).eps)
            )

            np.testing.assert_allclose(0, deriv/loss(raveled_fm), atol=1e-3, rtol=1e-3)

    def test_unpenalized_grad(self, random_rank4_ktensor):
        self.check_gradient(random_rank4_ktensor)
        
    def perturb_factor_matrix(self, factor_matrix, noise):
        fm_norm = np.linalg.norm(factor_matrix)
        factor_matrix_perturbed = (
            factor_matrix + 
            noise*(np.random.standard_normal(size=factor_matrix.shape)/fm_norm)
        )
        return factor_matrix_perturbed

    def test_minimum(self, random_rank4_ktensor, **kwargs):
        # Check if loss increases after pertubation
        X = random_rank4_ktensor.construct_tensor()
        wrong_decomposition = KruskalTensor.random_init(
            random_rank4_ktensor.shape,
            rank=random_rank4_ktensor.rank
        )

        for mode in range(3):
            sub_problem = self.SubProblem(mode=mode, **kwargs)
            sub_problem.update_decomposition(X, wrong_decomposition)
            factor_matrix = wrong_decomposition.factor_matrices[mode]
            fm_norm = np.linalg.norm(factor_matrix)

            for noise in range(1, 11):
                noise /= 100
                factor_matrix_perturbed = self.perturb_factor_matrix(factor_matrix, noise)
                
                def loss(x):
                    factor_matrices = [
                        fm for fm in wrong_decomposition.factor_matrices
                    ]
                    factor_matrices[mode] = x
                    estimated = np.einsum('ir, jr, kr -> ijk', *factor_matrices)
                    return np.linalg.norm(estimated - X)**2
                
                assert loss(factor_matrix) <= loss(factor_matrix_perturbed)


class TestRLSSubproblem(BaseTestSubproblem):
    SubProblem = RLS

    def test_non_negative_grad(self, random_nonnegative_rank4_ktensor):
        self.check_gradient(random_nonnegative_rank4_ktensor, non_negativity=True)

class TestADMMSubproblem(BaseTestSubproblem):
    # TODO: Check KKT instead of gradient
    SubProblem = ADMMSubproblem
    def test_minimum(self, random_nonnegative_rank4_ktensor):
        super().test_minimum(random_nonnegative_rank4_ktensor, rho=1)

    def test_unpenalized_grad(self, random_nonnegative_rank4_ktensor):
        self.check_gradient(random_nonnegative_rank4_ktensor, 
                            non_negativity=True,
                            rho=1)

    def test_non_negative_grad(self, random_nonnegative_rank4_ktensor):
        self.check_gradient(random_nonnegative_rank4_ktensor, 
                            non_negativity=True,
                            rho=1)

    def perturb_factor_matrix(self, factor_matrix, noise):
        fm_norm = np.linalg.norm(factor_matrix)
        factor_matrix_perturbed = (
            factor_matrix + 
            noise*(np.random.standard_normal(size=factor_matrix.shape)/fm_norm)
        )
        return np.maximum(0, factor_matrix_perturbed)

class BaseTestParafac2Subproblem():
    SubProblem = None

    @pytest.fixture
    def random_rank4_parafac2_tensor(self):
        return Parafac2Tensor.random_init(sizes=[30, 40, 50], rank=4)

    def test_B_grad(self, random_rank4_parafac2_tensor):
        X = random_rank4_parafac2_tensor.construct_tensor()
        A = random_rank4_parafac2_tensor.A 
        blueprint_B = random_rank4_parafac2_tensor.blueprint_B
        C = random_rank4_parafac2_tensor.C

        ktensor = KruskalTensor([A, blueprint_B, C])
        projected_X = ktensor.construct_tensor()

        wrong_decomposition = Parafac2Tensor.random_init(
            sizes=random_rank4_parafac2_tensor.shape,
            rank=random_rank4_parafac2_tensor.rank
        )

        
        sub_problem = Parafac2RLS()
        sub_problem.update_decomposition(
            X, wrong_decomposition, projected_X, should_update_projections=False
        )
        new_B = wrong_decomposition.blueprint_B

        def loss(x):
            B = x.reshape(wrong_decomposition.blueprint_B.shape)
            factor_matrices = [wrong_decomposition.A, B, wrong_decomposition.C]
            estimated = np.einsum('ir, jr, kr -> ijk', *factor_matrices)
            return np.linalg.norm(estimated - projected_X)**2

        deriv = approx_fprime(new_B.ravel(), loss, epsilon=np.sqrt(np.finfo(float).eps))
        assert np.allclose(deriv, 0, atol=1e-4, rtol=1e-4)

    def test_projected_X(self, random_rank4_parafac2_tensor):
        X = random_rank4_parafac2_tensor.construct_slices()
        A = random_rank4_parafac2_tensor.A 
        blueprint_B = random_rank4_parafac2_tensor.blueprint_B
        C = random_rank4_parafac2_tensor.C

        ktensor = KruskalTensor([A, blueprint_B, C])
        projected_X = ktensor.construct_tensor()

        wrong_decomposition = Parafac2Tensor.random_init(
            sizes=random_rank4_parafac2_tensor.shape,
            rank=random_rank4_parafac2_tensor.rank
        )

        sub_problem = Parafac2RLS()
        sub_problem.update_decomposition(
            X, wrong_decomposition, projected_X, should_update_projections=True
        )

        for projection in wrong_decomposition.projection_matrices:
            assert np.allclose(projection.T@projection, np.identity(projection.shape[1]))

    def test_update_status(self):
        pass

class TestParafac2RLSSubproblem(BaseTestParafac2Subproblem):
    SubProblem = Parafac2RLS
    
class TestParafac2ADMMSubproblem(BaseTestParafac2Subproblem):
    SubProblem = Parafac2ADMM

    def test_smoothness_prox_grad(self, random_rank4_parafac2_tensor):
        num_nodes = random_rank4_parafac2_tensor.shape[1]
        B = np.array(random_rank4_parafac2_tensor.B).copy()

        L = np.zeros((num_nodes, num_nodes))
        for node in range(num_nodes):
            L[(node - 1) % num_nodes, node] -= 1
            L[node, node] += 2
            L[(node + 1) % num_nodes, node] -= 1
        rho = 2
        smooth_admm = self.SubProblem(l2_similarity=0*L, rho=rho,)
        B2 = np.array(
            [
                smooth_admm.constraint_prox(Bk, random_rank4_parafac2_tensor, k) 
                for k, Bk in enumerate(B)
            ]
        )

        def loss(x):
            x = x.reshape(B.shape)
            return smooth_admm.regulariser(x) + (rho/2)*np.sum((x - B)**2)
        
        deriv = approx_fprime(B2.ravel(), loss, 1e-10)
        assert np.linalg.norm(deriv, np.inf) < 1e-5

class TestBlockParafac2:
    @pytest.fixture
    def random_rank4_parafac2_tensor(self):
        return Parafac2Tensor.random_init(sizes=[30, 40, 50], rank=4)

    @pytest.fixture
    def random_nonnegative_rank4_ktensor(self):
        return KruskalTensor.random_init([30, 40, 50], 4, random_method='uniform')
    

    def test_parafac2_decomposition(self, random_nonnegative_rank4_ktensor):
        X = random_nonnegative_rank4_ktensor.construct_tensor()
        pf2 = BlockParafac2(
            rank=4,
            sub_problems=[
                RLS(mode=0),
                Parafac2RLS(),
                RLS(mode=2, non_negativity=False),
            ],
            convergence_tol=1e-6
        )
        pf2.fit(X)

        assert pf2.explained_variance > (1-1e-3)

class TestADMMParafac2(TestBlockParafac2):
    def test_parafac2_decomposition(self, random_nonnegative_rank4_ktensor):
        X = random_nonnegative_rank4_ktensor.construct_tensor()
        pf2 = BlockParafac2(
            rank=4,
            sub_problems=[
                ADMMSubproblem(mode=0, rho=1),
                Parafac2RLS(),
                ADMMSubproblem(mode=2, non_negativity=False, rho=1),
            ],
            convergence_tol=1e-8
        )
        pf2.fit(X)

        assert pf2.explained_variance > (1-1e-3)

    def test_parafac2_decomposition_non_negative_A_and_C(self, random_nonnegative_rank4_ktensor):
        X = random_nonnegative_rank4_ktensor.construct_tensor()
        pf2 = BlockParafac2(
            rank=4,
            sub_problems=[
                ADMMSubproblem(mode=0, non_negativity=True, rho=1),
                Parafac2RLS(),
                ADMMSubproblem(mode=2, non_negativity=True, rho=1),
            ],
            convergence_tol=1e-8
        )
        pf2.fit(X)

        assert pf2.explained_variance > (1-1e-3)
    pass
