from pathlib import Path
import tempfile

import h5py
import pytest
import numpy as np
from .. import cp
from ... import base
from ... import metrics
# Husk: Test at weights og factors endres inplace

np.random.seed(0)

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
    
    def test_store_and_load_from_checkpoint(self, rank4_kruskal_tensor):
        max_its = 20
        checkpoint_period = 5
        X = rank4_kruskal_tensor.construct_tensor()
        with tempfile.TemporaryDirectory() as tempfolder:
            checkpoint_name = f'{tempfolder}/checkpoint.h5'
            cp_als = cp.CP_ALS(
                4, max_its=max_its, convergence_tol=1e-20, checkpoint_period=checkpoint_period,
                checkpoint_name=checkpoint_name, print_frequency=-5
            )
            decomposition = cp_als.fit_transform(X)

            assert Path(checkpoint_name).is_file()
            
            with h5py.File(checkpoint_name) as h5:
                for i in range(max_its):
                    if (i+1) % checkpoint_period == 0:
                        assert f'checkpoint_{i:05d}' in h5


            cp_als2 = cp.CP_ALS(
                4, max_its=100, convergence_tol=1e-20, init='from_checkpoint'
            )
            cp_als2._init_fit(X, 100, checkpoint_name)
            for fm1, fm2 in zip(cp_als.decomposition.factor_matrices, cp_als2.decomposition.factor_matrices):
                assert np.allclose(fm1, fm2)
            
            assert np.allclose(cp_als.decomposition.weights, cp_als2.decomposition.weights)
        
