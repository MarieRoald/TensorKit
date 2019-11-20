import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from tenkit import metrics
from tenkit.decomposition import cmtf, cp, decompositions

np.random.seed(0)

#TODO: test coupling on different modes and more than one mode

class TestCMTFALS:
    @pytest.fixture
    def rank4_kruskal_tensor(self):
        ktensor = decompositions.KruskalTensor.random_init((30, 40, 50), rank=4)
        ktensor.normalize_components()
        return ktensor

    @pytest.fixture
    def rank4_coupled_matrix_factors(self):
        A = np.random.standard_normal((30, 4))
        V = np.random.standard_normal((45,4))

        return A, V

    def test_rank4_cmtf_with_one_coupled_matrix(self, rank4_kruskal_tensor, rank4_coupled_matrix_factors):
        X = rank4_kruskal_tensor.construct_tensor()
        A, V = rank4_coupled_matrix_factors
        A = rank4_kruskal_tensor.factor_matrices[0]
        Y = A @ V.T

        cmtf_decomposer = cmtf.CMTF_ALS(4, max_its=2000, convergence_tol=0e-10, print_frequency=100)

        estimated_ktensor, estimated_Y_factors = cmtf_decomposer.fit_transform(X, [Y], [0])
        estimated_X = estimated_ktensor.construct_tensor()
        estimated_A, estimated_V = estimated_Y_factors[0]

        estimated_Y = estimated_A @ estimated_V.T
        assert np.allclose(X, estimated_X)
        assert np.allclose(Y, estimated_Y)

        assert metrics.factor_match_score(
            [A, V], [estimated_A, estimated_V], weight_penalty=False
        )[0] > 1-1e-9

        assert metrics.factor_match_score(
            rank4_kruskal_tensor.factor_matrices, estimated_ktensor.factor_matrices, weight_penalty=False
        )[0] > 1-1e-9
