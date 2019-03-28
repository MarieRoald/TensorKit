import pytest
import numpy as np
from .. import parafac2
from ... import base
from ... import metrics
# Husk: Test at weights og factors endres inplace
# TODO: test factor match score

class TestCPALS:
    @pytest.fixture
    def rank4_parafac2_tensor(self):
        pf2tensor = base.Parafac2Tensor.random_init((30, 40, 50), rank=4)
        #pf2tensor.normalize_components()
        return pf2tensor
    
    def test_rank4_decomposition(self, rank4_parafac2_tensor):
        X = (rank4_parafac2_tensor.construct_tensor())
        parafac2_als = parafac2.Parafac2_ALS(4, max_its=10000, convergence_tol=1e-10, print_frequency=1000)
        estimated_pf2tensor = parafac2_als.fit_transform(X)
        estimated_X = estimated_pf2tensor.construct_tensor()

        assert np.allclose(X, estimated_X, rtol=1e-5, atol=1)