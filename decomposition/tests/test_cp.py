import pytest
import numpy as np
from .. import cp
from ... import base
from ... import metrics
# Husk: Test at weights og factors endres inplace


class TestCPALS:
    @pytest.fixture
    def rank4_kruskal_tensor(self):
        ktensor = base.KruskalTensor.random_init((30, 40, 50), rank=4)
        ktensor.normalize_components()
        return ktensor
    
    def test_rank4_decomposition(self, rank4_kruskal_tensor):
        X = rank4_kruskal_tensor.construct_tensor()
        cp_als = cp.CP_ALS(4, max_its=10000, convergence_tol=1e-10)
        estimated_ktensor = cp_als.fit_transform(X)
        estimated_ktensor.normalize_components()

        assert np.allclose(X, estimated_ktensor.construct_tensor())
        assert metrics.factor_match_score(
            rank4_kruskal_tensor.factor_matrices, estimated_ktensor.factor_matrices
        )[0] > 1-1e-10