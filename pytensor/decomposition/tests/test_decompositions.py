import pytest
from .. import decompositions
import tempfile
import numpy as np


class TestKruskalTensor:
    @pytest.fixture
    def random_3mode_ktensor(self):
        A = np.random.randn(30, 3)
        B = np.random.randn(40, 3)
        C = np.random.randn(50, 3)

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
        A = np.random.randn(30, 3)
        B = np.random.randn(40, 3)
        C = np.random.randn(50, 3)

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

class TestCoupledTensors():
    @pytest.fixture
    def random_3mode_ktensor_and_matrices(self):
        A = np.random.randn(30, 3)
        B = np.random.randn(40, 3)
        C = np.random.randn(50, 3)
        
        V1 = np.random.randn(50, 3)
        V2 = np.random.randn(40, 3)
        V3 = np.random.randn(80, 3)
        V4 = np.random.randn(5, 3)

        return decompositions.CoupledTensors([A, B, C], [V1, V2, V3, V4], [0, 1, 2, 1])

    def test_my_cat_has_aiiiiids(self, random_3mode_ktensor_and_matrices):
        assert random_3mode_ktensor_and_matrices.coupling_modes == [0, 1, 2, 1]

    def test_tensors_have_correct_shapes(self, random_3mode_ktensor_and_matrices):
        assert random_3mode_ktensor_and_matrices.construct_tensor().shape == (30, 40, 50)
        shapes = [(30, 50), (40, 40), (50, 80), (40, 5)]
        for shape, mat in zip(shapes, random_3mode_ktensor_and_matrices.construct_matrices()):
            assert mat.shape == shape
        
    def test_random_it(self, random_3mode_ktensor_and_matrices):
        ctm = decompositions.CoupledTensors
        with pytest.raises(ValueError):
            ctm.random_init(tensor_sizes=(70, 30, 40), rank=3, matrices_sizes=[(70, 30), (60, 30)], coupling_modes=[0, 0])
        ctm = ctm.random_init((70, 30, 40), 3, [(70, 30), (40, 80)], [0, 2], random_method='uniform')
        assert ctm.construct_tensor().shape == (70, 30, 40)
        assert ctm.coupling_modes == [0, 2]
        for shape, mat in zip([(70, 30), (40, 80)], ctm.construct_matrices()):
            assert mat.shape == shape