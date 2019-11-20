import itertools
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from tenkit import base, metrics
from tenkit.decomposition import decompositions, parafac2

from .test_utils import ensure_monotonicity

# Husk: Test at weights og factors endres inplace
# TODO: test factor match score

class TestParafac2ALS:
    @pytest.fixture
    def rank4_parafac2_tensor(self):
        pf2tensor = decompositions.Parafac2Tensor.random_init((30, 40, 50), rank=4)
        #pf2tensor.normalize_components()
        return pf2tensor
    
    def test_rank4_decomposition_cp_init(self, rank4_parafac2_tensor):
        X = (rank4_parafac2_tensor.construct_tensor())
        parafac2_als = parafac2.Parafac2_ALS(4, max_its=1000, convergence_tol=1e-10, print_frequency=1000, init='cp')
        estimated_pf2tensor = parafac2_als.fit_transform(X)
        estimated_X = estimated_pf2tensor.construct_tensor()

        assert np.allclose(X, estimated_X, rtol=1e-5, atol=1)

    def test_rank4_decomposition(self, rank4_parafac2_tensor):
        X = (rank4_parafac2_tensor.construct_tensor())
        parafac2_als = parafac2.Parafac2_ALS(4, max_its=1000, convergence_tol=1e-10, print_frequency=1000)
        estimated_pf2tensor = parafac2_als.fit_transform(X)
        estimated_X = estimated_pf2tensor.construct_tensor()

        assert np.allclose(X, estimated_X, rtol=1e-5, atol=1)

    def test_rank4_non_negative_decomposition(self, rank4_parafac2_tensor):
        X = (rank4_parafac2_tensor.construct_tensor())
        parafac2_als = parafac2.Parafac2_ALS(4, non_negativity_constraints=[False, True, True], max_its=1000, convergence_tol=1e-10, print_frequency=1000)
        for constraint in itertools.product([True, False], repeat=3):
            parafac2_als.non_negativity_constraints = constraint
            estimated_pf2tensor = parafac2_als.fit_transform(X)
            estimated_X = estimated_pf2tensor.construct_tensor()

            assert np.allclose(X, estimated_X, rtol=1e-5, atol=1)

    def test_rank4_monotone_convergence_projections(self, rank4_parafac2_tensor):
        X = (rank4_parafac2_tensor.construct_tensor())
        parafac2_als = parafac2.Parafac2_ALS(4, max_its=100, convergence_tol=1e-10, print_frequency=1000)
        parafac2_als._update_projection_matrices = ensure_monotonicity(
            parafac2_als,
            '_update_projection_matrices',
            'SSE',
            atol=1e-8,
            rtol=1e-4
        )
        parafac2_als.fit_transform(X)
 
    def test_rank4_monotone_convergence(self, rank4_parafac2_tensor):
        X = (rank4_parafac2_tensor.construct_tensor())
        parafac2_als = parafac2.Parafac2_ALS(4, max_its=100, convergence_tol=1e-10, print_frequency=1000)
        parafac2_als._update_parafac2_factors = ensure_monotonicity(
            parafac2_als,
            '_update_parafac2_factors',
            'SSE',
            atol=1e-8,
            rtol=1e-4
        )
        parafac2_als.fit_transform(X)

    def assert_correlation(self, M1, M2):
        assert metrics.factor_match_score([M1], [M2], weight_penalty=False)[0] > 1 - 1e-3

    def test_rank4_decomposition_correlation(self, rank4_parafac2_tensor):
        X = (rank4_parafac2_tensor.construct_tensor())
        parafac2_als = parafac2.Parafac2_ALS(4, max_its=1000, convergence_tol=1e-10, print_frequency=1000)
        estimated_pf2tensor = parafac2_als.fit_transform(X)

        self.assert_correlation(rank4_parafac2_tensor.A, estimated_pf2tensor.A)
        for P1, P2 in zip(rank4_parafac2_tensor.projection_matrices, estimated_pf2tensor.projection_matrices):
            Bk = P1@rank4_parafac2_tensor.blueprint_B
            estimated_Bk = P2@estimated_pf2tensor.blueprint_B

            self.assert_correlation(Bk, estimated_Bk)

        self.assert_correlation(rank4_parafac2_tensor.C, estimated_pf2tensor.C)

    def test_rank4_non_negative_decomposition_correlation(self, rank4_parafac2_tensor):
        X = (rank4_parafac2_tensor.construct_tensor())
        parafac2_als = parafac2.Parafac2_ALS(4,  non_negativity_constraints=[False, False, True], max_its=1000, convergence_tol=1e-10, print_frequency=1000)
        
        for constraint in itertools.product([True, False], repeat=3):
            parafac2_als.non_negativity_constraints = constraint
            estimated_pf2tensor = parafac2_als.fit_transform(X)

            self.assert_correlation(rank4_parafac2_tensor.A, estimated_pf2tensor.A)
            for P1, P2 in zip(rank4_parafac2_tensor.projection_matrices, estimated_pf2tensor.projection_matrices):
                Bk = P1@rank4_parafac2_tensor.blueprint_B
                estimated_Bk = P2@estimated_pf2tensor.blueprint_B

                self.assert_correlation(Bk, estimated_Bk)

            self.assert_correlation(rank4_parafac2_tensor.C, estimated_pf2tensor.C)

    def test_store_and_load_from_checkpoint(self, rank4_parafac2_tensor):
        max_its = 20
        checkpoint_frequency = 5
        X = rank4_parafac2_tensor.construct_tensor()
        with tempfile.TemporaryDirectory() as tempfolder:
            checkpoint_path = f'{tempfolder}/checkpoint.h5'
            parafac2_decomposer = parafac2.Parafac2_ALS(
                4, max_its=max_its, convergence_tol=1e-20, checkpoint_frequency=checkpoint_frequency,
                checkpoint_path=checkpoint_path, print_frequency=-5
            )
            decomposition = parafac2_decomposer.fit_transform(X)

            assert Path(checkpoint_path).is_file()
            
            with h5py.File(checkpoint_path) as h5:
                for i in range(max_its):
                    if (i+1) % checkpoint_frequency == 0:
                        assert f'checkpoint_{i:05d}' in h5


            parafac2_decomposer2 = parafac2.Parafac2_ALS(
                4, max_its=100, convergence_tol=1e-20, init='from_checkpoint'
            )
            parafac2_decomposer2._init_fit(X, 100, checkpoint_path)

            assert np.allclose(parafac2_decomposer.decomposition.A, parafac2_decomposer2.decomposition.A)
            assert np.allclose(parafac2_decomposer.decomposition.C, parafac2_decomposer2.decomposition.C)

            assert np.allclose(parafac2_decomposer.decomposition.blueprint_B, parafac2_decomposer2.decomposition.blueprint_B)
                
            for P1, P2 in zip(parafac2_decomposer.decomposition.projection_matrices, parafac2_decomposer2.decomposition.projection_matrices):
                assert np.allclose(P1, P2)
