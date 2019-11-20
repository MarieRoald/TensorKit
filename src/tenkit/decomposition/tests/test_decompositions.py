import tempfile

import numpy as np
import pytest

from tenkit.decomposition import decompositions


class TestKruskalTensor:
    @pytest.fixture
    def random_3mode_ktensor(self):
        A = np.random.randn(30, 4)
        B = np.random.randn(40, 4)
        C = np.random.randn(50, 4)

        return decompositions.KruskalTensor([A, B, C])
    
    def test_load_tensor_loads_stored_tensor(self, random_3mode_ktensor):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = f'{tmpdir}/storedfactors.h5'
            random_3mode_ktensor.store(filename)
            
            loaded_tensor = decompositions.KruskalTensor.from_file(filename)
        
        assert np.allclose(loaded_tensor.weights, random_3mode_ktensor.weights)
        assert loaded_tensor.weights.shape == random_3mode_ktensor.weights.shape

        for fm, lfm in zip(random_3mode_ktensor.factor_matrices, loaded_tensor.factor_matrices):
            np.allclose(fm, lfm)

    def test_all_factor_matrices_must_have_same_size(self):
        A = np.random.randn(30, 5)
        B = np.random.randn(40, 5)
        C = np.random.randn(50, 6)

        with pytest.raises(ValueError):
            decompositions.KruskalTensor([A, B, C])
        
        B = np.random.randn(40, 6)
        with pytest.raises(ValueError):
            decompositions.KruskalTensor([A, B, C])
        
        A = np.random.randn(30, 6)
        B = np.random.randn(40, 5)
        with pytest.raises(ValueError):
            decompositions.KruskalTensor([A, B, C])
    
    def test_correct_size_of_tensor(self):
        A = np.random.randn(30, 4)
        B = np.random.randn(40, 4)
        C = np.random.randn(50, 4)

        ktensor = decompositions.KruskalTensor([A, B, C])
        assert ktensor.construct_tensor().shape == (30, 40, 50)
    
    def test_normalize_ktensor_doesnt_change_constructed_tensor(self, random_3mode_ktensor):
        unnormalized_tensor = random_3mode_ktensor.construct_tensor().copy()
        random_3mode_ktensor.normalize_components()
        assert np.allclose(unnormalized_tensor, random_3mode_ktensor.construct_tensor())
    
    def test_normalize_ktensor_normalizes_ktensor(self, random_3mode_ktensor):
        random_3mode_ktensor.normalize_components()
        units = np.ones(random_3mode_ktensor.rank)
        
        for factor_matrix in random_3mode_ktensor.factor_matrices:
            assert np.allclose(np.linalg.norm(factor_matrix, axis=0), units)
        
    def test_tensor_is_constructed_correctly(self, random_3mode_ktensor):
        tensor = random_3mode_ktensor.construct_tensor()
        A, B, C = random_3mode_ktensor.factor_matrices

        for i, matrix in enumerate(tensor):
            for j, vector in enumerate(matrix):
                for k, element in enumerate(vector):
                    assert abs(element - np.sum(A[i, :]*B[j, :]*C[k, :])) < 1e-8
    
    def test_sign_flip_yields_consistent_decomposition(self, random_3mode_ktensor):
        X = random_3mode_ktensor.construct_tensor()
        X += np.random.standard_normal(X.shape)*np.linalg.norm(X)*0.1
        signs = random_3mode_ktensor.get_signs(X)

        for rank in range(random_3mode_ktensor.rank):
            sign_product = 1
            for mode in range(3):
                sign_product *= signs[mode][rank]
            assert sign_product == 1
    
    def test_sign_flip_yields_same_sign_as_sign_score(self, random_3mode_ktensor):
        X = random_3mode_ktensor.construct_tensor()
        X += np.random.standard_normal(X.shape)*np.linalg.norm(X)*0.1
        signs = random_3mode_ktensor.get_signs(X)
        sign_scores = random_3mode_ktensor.get_sign_scores(X)

        for rank in range(random_3mode_ktensor.rank):
            sign_product = 1
            num_equal_signs = 0
            for mode in range(3):
                sign_product *= np.sign(sign_scores[mode][rank])
                num_equal_signs += np.sign(sign_scores[mode][rank]) == signs[mode][rank]
            
            if sign_product == -1:
                assert num_equal_signs == 2
            else:
                assert num_equal_signs == 3
        
    def test_get_single_component_decomposition(self, random_3mode_ktensor):
        factor_matrices = random_3mode_ktensor.factor_matrices
        for component in range(4):
            single_component_decomposition = (
                random_3mode_ktensor.get_single_component_decomposition(component)
            )
            for fm1, fm2 in zip(factor_matrices, single_component_decomposition.factor_matrices):
                assert np.allclose(fm1[:, component], fm2[:, 0])
            assert random_3mode_ktensor.weights[component] == single_component_decomposition.weights[0]
            assert single_component_decomposition.rank == 1

    

class TestEvolvingTensor:
    @pytest.fixture
    def uniform_evolving_tensor(self):
        A = np.random.randn(30, 5)
        B = [np.random.randn(40, 5) for _ in range(50)]
        C = np.random.randn(50, 5)

        return decompositions.EvolvingTensor(A, B, C)
    
    @pytest.fixture
    def nonuniform_evolving_tensor(self):
        A = np.random.randn(30, 5)
        B = [np.random.randn(np.random.randint(30, 50), 5) for _ in range(50)]
        C = np.random.randn(50, 5)
        
        return decompositions.EvolvingTensor(A, B, C, warning=False)
    
    def test_load_uniform_tensor_loads_stored_tensor(self, uniform_evolving_tensor):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = f'{tmpdir}/storedfactors.h5'
            uniform_evolving_tensor.store(filename)
            
            loaded_tensor = decompositions.EvolvingTensor.from_file(filename)
        
        assert np.allclose(loaded_tensor.A, uniform_evolving_tensor.A)
        assert np.allclose(loaded_tensor.C, uniform_evolving_tensor.C)

        for Bk, lBk in zip(uniform_evolving_tensor.B, loaded_tensor.B):
            np.allclose(Bk, lBk)
    
    def test_load_nonuniform_tensor_loads_stored_tensor(self, nonuniform_evolving_tensor):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = f'{tmpdir}/storedfactors.h5'
            nonuniform_evolving_tensor.store(filename)
            
            loaded_tensor = decompositions.EvolvingTensor.from_file(filename)
        
        assert np.allclose(loaded_tensor.A, nonuniform_evolving_tensor.A)
        assert np.allclose(loaded_tensor.C, nonuniform_evolving_tensor.C)

        for Bk, lBk in zip(nonuniform_evolving_tensor.B, loaded_tensor.B):
            np.allclose(Bk, lBk)

    def test_correct_size_of_uniform_tensor(self, uniform_evolving_tensor):
        shape = (
            uniform_evolving_tensor.A.shape[0],
            uniform_evolving_tensor.B[0].shape[0],
            uniform_evolving_tensor.C.shape[0]
        )
        assert uniform_evolving_tensor.construct_tensor().shape == shape
    
    def test_correct_size_of_nonuniform_tensor(self, nonuniform_evolving_tensor):
        shape = (
            nonuniform_evolving_tensor.A.shape[0],
            max(m.shape[0] for m in nonuniform_evolving_tensor.B),
            nonuniform_evolving_tensor.C.shape[0]
        )

        assert nonuniform_evolving_tensor.construct_tensor().shape == shape

    def test_uniform_tensor_is_constructed_correctly(self, uniform_evolving_tensor):
        A = uniform_evolving_tensor.A
        B = uniform_evolving_tensor.B
        C = uniform_evolving_tensor.C

        tensor = uniform_evolving_tensor.construct_tensor()


        for i, a_i in enumerate(A):
            for k, c_k in enumerate(C):
                for j, b_kj in enumerate(B[k]):
                    assert abs(tensor[i, j, k] - np.sum(a_i*b_kj*c_k)) < 1e-8

    def test_nonuniform_tensor_is_constructed_correctly(self, nonuniform_evolving_tensor):
        A = nonuniform_evolving_tensor.A
        B = nonuniform_evolving_tensor.B
        C = nonuniform_evolving_tensor.C

        tensor = nonuniform_evolving_tensor.construct_tensor()

        for i, a_i in enumerate(A):
            for k, c_k in enumerate(C):
                for j, b_kj in enumerate(B[k]):
                    assert abs(tensor[i, j, k] - np.sum(a_i*b_kj*c_k)) < 1e-8

    def test_uniform_tensor_slices_equals_constructed_tensor(self, uniform_evolving_tensor):
        tensor = uniform_evolving_tensor.construct_tensor()
        for k, slice_ in enumerate(uniform_evolving_tensor.construct_slices()):
            assert np.allclose(tensor[:, :, k], slice_)

    def test_nonuniform_tensor_slices_equals_constructed_tensor(self, nonuniform_evolving_tensor):
        tensor = nonuniform_evolving_tensor.construct_tensor()
        for k, slice_ in enumerate(nonuniform_evolving_tensor.construct_slices()):
            assert np.allclose(tensor[:, :slice_.shape[1], k], slice_)

    def test_nonuniform_tensor_is_padded_with_zeros(self, nonuniform_evolving_tensor):
        tensor = nonuniform_evolving_tensor.construct_tensor()
        for k, slice_ in enumerate(nonuniform_evolving_tensor.construct_slices()):
            assert np.allclose(tensor[:, slice_.shape[1]:, k], 0)
    
    def test_nonuniform_slices_has_correct_size(self, nonuniform_evolving_tensor):
        m = nonuniform_evolving_tensor.A.shape[0]
        B = nonuniform_evolving_tensor.B

        for Bk, slice_ in zip(B, nonuniform_evolving_tensor.construct_slices()):
            n = Bk.shape[0]

            assert slice_.shape == (m, n)


class TestParafac2Tensor(TestEvolvingTensor):
    @pytest.fixture
    def uniform_evolving_tensor(self):
        A = np.random.randn(30, 5)
        B_blueprint = np.identity(5)*0.2 + 0.8
        projection_matrices = [np.linalg.qr(np.random.randn(40, 5))[0] for _ in range(50)]
        C = np.random.randn(50, 5)

        return decompositions.Parafac2Tensor(A, B_blueprint, C, projection_matrices)
    
    @pytest.fixture
    def nonuniform_evolving_tensor(self):
        A = np.random.randn(30, 5)
        J_ks = [np.random.randint(30, 50) for _ in range(50)]
        projection_matrices = [np.linalg.qr(np.random.randn(J, 5))[0] for J in J_ks]
        B_blueprint = np.identity(5)*0.2 + 0.8
        C = np.random.randn(50, 5)
        
        return decompositions.Parafac2Tensor(A, B_blueprint, C, projection_matrices, warning=False)
    
    def test_parafac2_constant_phi(self, uniform_evolving_tensor):
        Bk = uniform_evolving_tensor.B[0]
        phi = Bk.T@Bk
        for Bk in uniform_evolving_tensor.B:
            newphi = Bk.T@Bk
            assert np.allclose(phi, newphi)
    
    def test_load_uniform_tensor_loads_stored_tensor(self, uniform_evolving_tensor):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = f'{tmpdir}/storedfactors.h5'
            uniform_evolving_tensor.store(filename)
            
            loaded_tensor = decompositions.Parafac2Tensor.from_file(filename)
        
        assert np.allclose(loaded_tensor.A, uniform_evolving_tensor.A)
        assert np.allclose(loaded_tensor.C, uniform_evolving_tensor.C)
        assert np.allclose(loaded_tensor.blueprint_B, uniform_evolving_tensor.blueprint_B)

        for Pk, lPk in zip(uniform_evolving_tensor.projection_matrices, loaded_tensor.projection_matrices):
            np.allclose(Pk, lPk)
    
    def test_load_nonuniform_tensor_loads_stored_tensor(self, nonuniform_evolving_tensor):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = f'{tmpdir}/storedfactors.h5'
            nonuniform_evolving_tensor.store(filename)
            
            loaded_tensor = decompositions.Parafac2Tensor.from_file(filename)
        
        assert np.allclose(loaded_tensor.A, nonuniform_evolving_tensor.A)
        assert np.allclose(loaded_tensor.C, nonuniform_evolving_tensor.C)
        assert np.allclose(loaded_tensor.blueprint_B, nonuniform_evolving_tensor.blueprint_B)

        for Pk, lPk in zip(nonuniform_evolving_tensor.projection_matrices, loaded_tensor.projection_matrices):
            np.allclose(Pk, lPk)
