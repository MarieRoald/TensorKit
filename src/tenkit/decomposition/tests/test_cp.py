from pathlib import Path
import tempfile
from functools import wraps
import itertools

import h5py
import pytest
import numpy as np
from .test_utils import ensure_monotonicity
from tenkit.decomposition import cp
from tenkit.decomposition import decompositions
from tenkit import metrics
# Husk: Test at weights og factors endres inplace


class TestCPALS:
    @pytest.fixture
    def rank4_kruskal_tensor(self):
        ktensor = decompositions.KruskalTensor.random_init((30, 40, 50), rank=4)
        ktensor.normalize_components()
        return ktensor

    @pytest.fixture
    def nonnegative_rank4_kruskal_tensor(self):
        ktensor = decompositions.KruskalTensor.random_init((30, 40, 50), rank=4)
        ktensor.normalize_components()
        for i, fm in enumerate(ktensor.factor_matrices):
            ktensor.factor_matrices[i][...] = np.abs(fm)

        return ktensor
    
    def check_decomposition(self, kruskal_tensor, *, test_closeness=True, test_fms=True, **additional_params):

        X = kruskal_tensor.construct_tensor()
        cp_als = cp.CP_ALS(4, max_its=1000, convergence_tol=1e-10, **additional_params)
        estimated_ktensor = cp_als.fit_transform(X)
        estimated_ktensor.normalize_components()

        if test_closeness:
            assert np.allclose(X, estimated_ktensor.construct_tensor())
        if test_fms:
            assert metrics.factor_match_score(
                kruskal_tensor.factor_matrices, estimated_ktensor.factor_matrices
            )[0] > 1-1e-7
        
    def test_rank4_decomposition(self, rank4_kruskal_tensor):
        self.check_decomposition(rank4_kruskal_tensor)

    def test_rank4_nonnegative_decomposition(self, nonnegative_rank4_kruskal_tensor):
        self.check_decomposition(nonnegative_rank4_kruskal_tensor, non_negativity_constraints=[True, True, True])
    
    def test_rank4_l2_regularisation(self, rank4_kruskal_tensor):
        pass
        # self.check_decomposition(rank4_kruskal_tensor, ridge_penalties=[1e-5, 1e-5, 1e-5])

    def check_monotone_convergence(self, kruskal_tensor, **additional_params):
        X = kruskal_tensor.construct_tensor()
        cp_als = cp.CP_ALS(4, max_its=100, convergence_tol=1e-10, **additional_params)
        cp_als._update_als_factor = ensure_monotonicity(
            cp_als,
            '_update_als_factor',
            'loss',
            tol=1e-20
        )
        cp_als.fit_transform(X)

    def test_rank4_monotone_convergence(self, rank4_kruskal_tensor):
        self.check_monotone_convergence(rank4_kruskal_tensor)
    
    def test_rank4_nonnegative_monotone_convergence(self, nonnegative_rank4_kruskal_tensor):
        for constraints in itertools.product([True, False], repeat=3):
            self.check_monotone_convergence(nonnegative_rank4_kruskal_tensor, non_negativity_constraints=constraints)
    
    def test_rank4_ridge_monotone_convergence(self, rank4_kruskal_tensor):
        self.check_monotone_convergence(rank4_kruskal_tensor, ridge_penalties=[0.01, 0.01, 0.01])
    
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
        
