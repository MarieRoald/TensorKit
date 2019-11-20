import numpy as np
import pytest

from tenkit import metrics
from tenkit.decomposition.decompositions import Parafac2Tensor


class TestCoreConsistency:

    @pytest.fixture
    def random_parafac2_tensor(self):
        return Parafac2Tensor.random_init((10, [20]*30, 30), 5)

    def test_parafac2_core_consistency_perfect_decomposition(self, random_parafac2_tensor):
        X = random_parafac2_tensor.construct_slices()
        P = random_parafac2_tensor.projection_matrices
        A = random_parafac2_tensor.A 
        B = random_parafac2_tensor.blueprint_B
        C = random_parafac2_tensor.C
        
        cc = np.asscalar(metrics.core_consistency_parafac2(X, P, A, B, C))
        assert abs(cc-100) < 1e-10
