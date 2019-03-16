import pytest
import numpy as np
from .. import base


class TestKruskalTensor:
    @pytest.fixture
    def random_3mode_ktensor(self):
        A = np.random.randn(30, 3)
        B = np.random.randn(40, 3)
        C = np.random.randn(50, 3)

        return base.KruskalTensor([A, B, C])

    def test_all_factor_matrices_must_have_same_size(self):
        A = np.random.randn(30, 5)
        B = np.random.randn(40, 5)
        C = np.random.randn(50, 6)

        with pytest.raises(ValueError):
            base.KruskalTensor([A, B, C])
        
        B = np.random.randn(40, 6)
        with pytest.raises(ValueError):
            base.KruskalTensor([A, B, C])
        
        A = np.random.randn(30, 6)
        B = np.random.randn(40, 5)
        with pytest.raises(ValueError):
            base.KruskalTensor([A, B, C])
    
    def test_correct_size_of_tensor(self):
        A = np.random.randn(30, 3)
        B = np.random.randn(40, 3)
        C = np.random.randn(50, 3)

        ktensor = base.KruskalTensor([A, B, C])
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
    
    