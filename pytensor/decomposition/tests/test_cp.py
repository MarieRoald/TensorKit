from pathlib import Path
import tempfile
from functools import wraps
import itertools

import h5py
import pytest
import numpy as np
from .test_utils import ensure_monotonicity
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

    @pytest.fixture
    def nonnegative_rank4_kruskal_tensor(self):
        ktensor = base.KruskalTensor.random_init((30, 40, 50), rank=4)
        ktensor.normalize_components()
        for i, fm in enumerate(ktensor.factor_matrices):
            ktensor.factor_matrices[i][...] = np.abs(fm)

        return ktensor
    
    def test_rank4_decomposition(self, rank4_kruskal_tensor):
        X = rank4_kruskal_tensor.construct_tensor()
        cp_als = cp.CP_ALS(4, max_its=1000, convergence_tol=1e-10)
        estimated_ktensor = cp_als.fit_transform(X)
        estimated_ktensor.normalize_components()

        assert np.allclose(X, estimated_ktensor.construct_tensor())
        assert metrics.factor_match_score(
            rank4_kruskal_tensor.factor_matrices, estimated_ktensor.factor_matrices
        )[0] > 1-1e-7

    def test_rank4_nonnegative_decomposition(self, nonnegative_rank4_kruskal_tensor):
        X = nonnegative_rank4_kruskal_tensor.construct_tensor()
        cp_als = cp.CP_ALS(4, max_its=1000, convergence_tol=1e-10, non_negativity_constraints=[True, True, True])
        estimated_ktensor = cp_als.fit_transform(X)
        estimated_ktensor.normalize_components()

        assert np.allclose(X, estimated_ktensor.construct_tensor())
        assert metrics.factor_match_score(
            nonnegative_rank4_kruskal_tensor.factor_matrices, estimated_ktensor.factor_matrices
        )[0] > 1-1e-7
    
    def test_rank4_monotone_convergence(self, rank4_kruskal_tensor):
        X = rank4_kruskal_tensor.construct_tensor()
        cp_als = cp.CP_ALS(4, max_its=100, convergence_tol=1e-10)
        cp_als._update_als_factor = ensure_monotonicity(
            cp_als,
            '_update_als_factor',
            'MSE',
            tol=1e-16
        )
        cp_als.fit_transform(X)
    
    def test_rank4_nonnegative_monotone_convergence(self, nonnegative_rank4_kruskal_tensor):
        X = nonnegative_rank4_kruskal_tensor.construct_tensor()
        cp_als = cp.CP_ALS(4, max_its=100, convergence_tol=1e-10, non_negativity_constraints=[True, True, True])
        cp_als._update_als_factor = ensure_monotonicity(
            cp_als,
            '_update_als_factor',
            'MSE',
            tol=1e-16
        )
        for constraints in itertools.product([True, False], repeat=3):
            cp_als.non_negativity_constraints = constraints
            cp_als.fit_transform(X)
    
    def test_rank4_regularised_monotone_convergence(self, rank4_kruskal_tensor):
        X = rank4_kruskal_tensor.construct_tensor()
        tik = [s*np.identity(s) for s in X.shape]
        tik[-1] = None

        cp_als = cp.CP_ALS(4, max_its=100, convergence_tol=1e-10, tikhonov_matrices=tik)
        cp_als._update_als_factor = ensure_monotonicity(
            cp_als,
            '_update_als_factor',
            'regularised_loss',
            tol=1e-16,
        )
        cp_als.fit_transform(X)
    
    def test_store_and_load_from_checkpoint(self, rank4_kruskal_tensor):
        max_its = 20
        checkpoint_frequency = 5
        X = rank4_kruskal_tensor.construct_tensor()
        with tempfile.TemporaryDirectory() as tempfolder:
            checkpoint_path = f'{tempfolder}/checkpoint.h5'
            cp_als = cp.CP_ALS(
                4, max_its=max_its, convergence_tol=1e-20, checkpoint_frequency=checkpoint_frequency,
                checkpoint_path=checkpoint_path, print_frequency=-5
            )
            decomposition = cp_als.fit_transform(X)

            assert Path(checkpoint_path).is_file()
            
            with h5py.File(checkpoint_path) as h5:
                for i in range(max_its):
                    if (i+1) % checkpoint_frequency == 0:
                        assert f'checkpoint_{i:05d}' in h5


            cp_als2 = cp.CP_ALS(
                4, max_its=100, convergence_tol=1e-20, init='from_checkpoint'
            )
            cp_als2._init_fit(X, 100, checkpoint_path)
            for fm1, fm2 in zip(cp_als.decomposition.factor_matrices, cp_als2.decomposition.factor_matrices):
                assert np.allclose(fm1, fm2)
            
            assert np.allclose(cp_als.decomposition.weights, cp_als2.decomposition.weights)
        
