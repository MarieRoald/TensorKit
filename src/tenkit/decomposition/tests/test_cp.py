import itertools
import tempfile
from functools import wraps
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.optimize import check_grad

from tenkit import base, metrics
from tenkit.decomposition import cp, decompositions

from .test_utils import ensure_monotonicity

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

    @pytest.fixture
    def orthogonal_rank4_kruskal_tensor(self):
        ktensor = decompositions.KruskalTensor.random_init((30, 40, 50), rank=4)
        ktensor.normalize_components()
        ktensor.factor_matrices[0] = np.linalg.qr(ktensor.factor_matrices[0])[0]

        return ktensor
    
    def check_decomposition(self, kruskal_tensor, *, test_closeness=True, test_fms=True, **additional_params):
        X = kruskal_tensor.construct_tensor()
        cp_als = cp.CP_ALS(4, max_its=1000, convergence_tol=1e-10, **additional_params)
        estimated_ktensor = cp_als.fit_transform(X)
        estimated_ktensor.normalize_components()

        if test_closeness:
            assert np.linalg.norm(X - estimated_ktensor.construct_tensor())**2/np.linalg.norm(X) < 1e-5
        if test_fms:
            assert metrics.factor_match_score(
                kruskal_tensor.factor_matrices, estimated_ktensor.factor_matrices
            )[0] > 1-1e-5
        
    def test_rank4_decomposition(self, rank4_kruskal_tensor):
        self.check_decomposition(rank4_kruskal_tensor)

    def test_rank4_nonnegative_decomposition(self, nonnegative_rank4_kruskal_tensor):
        self.check_decomposition(nonnegative_rank4_kruskal_tensor, non_negativity_constraints=[True, True, True])
    
    def test_rank4_orthogonal_decomposition(self, orthogonal_rank4_kruskal_tensor):
        self.check_decomposition(orthogonal_rank4_kruskal_tensor, orthonormality_constraints=[True, False, False])
    
    #def test_rank4_l2_regularisation(self, rank4_kruskal_tensor):
    #    pass
        # self.check_decomposition(rank4_kruskal_tensor, ridge_penalties=[1e-5, 1e-5, 1e-5])

    def check_monotone_convergence(self, kruskal_tensor, **additional_params):
        X = kruskal_tensor.construct_tensor()
        cp_als = cp.CP_ALS(4, max_its=100, convergence_tol=1e-10, **additional_params)
        cp_als._update_als_factor = ensure_monotonicity(
            cp_als,
            '_update_als_factor',
            'loss',
            atol=1e-8,
            rtol=1e-4,
        )
        cp_als.fit_transform(X)

    def test_rank4_monotone_convergence(self, rank4_kruskal_tensor):
        self.check_monotone_convergence(rank4_kruskal_tensor)
    
    def test_rank4_orthogonal_monotone_convergence(self, orthogonal_rank4_kruskal_tensor):
        self.check_monotone_convergence(orthogonal_rank4_kruskal_tensor, orthonormality_constraints=[True, False, False])
            
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
                4, max_its=max_its, convergence_tol=-1, rel_loss_tol=-1,
                checkpoint_frequency=checkpoint_frequency,
                checkpoint_path=checkpoint_path, print_frequency=-5
            )
            decomposition = cp_als.fit_transform(X)

            assert Path(checkpoint_path).is_file()
            
            with h5py.File(checkpoint_path) as h5:
                for i in range(max_its):
                    if (i+1) % checkpoint_frequency == 0:
                        assert f'checkpoint_{i:05d}' in h5


            cp_als2 = cp.CP_ALS(
                4, max_its=100, convergence_tol=-1, rel_loss_tol=-1, init='from_checkpoint'
            )
            cp_als2._init_fit(X, 100, checkpoint_path)
            for fm1, fm2 in zip(cp_als.decomposition.factor_matrices, cp_als2.decomposition.factor_matrices):
                assert np.allclose(fm1, fm2)
            
            assert np.allclose(cp_als.decomposition.weights, cp_als2.decomposition.weights)


        
class TestCPOPT:
    CP = cp.CP_OPT
    convergence_tol = 1e-20

    @pytest.fixture
    def rank4_kruskal_tensor(self):
        ktensor = decompositions.KruskalTensor.random_init((30, 40, 50), rank=4)
        ktensor.normalize_components()
        return ktensor
    
    @pytest.fixture
    def smooth_rank4_ktensor(self, rank4_kruskal_tensor):
        A, B, C = rank4_kruskal_tensor
        for i, prev_B in enumerate(B[1:]):
            B[i] = prev_B + np.linalg.norm(prev_B)*np.random.standard_normal(prev_B.shape)
        B /= np.linalg.norm(B, axis=0)

        return decompositions.KruskalTensor((A, B, C))

    @pytest.fixture
    def laplacian(self, smooth_rank4_ktensor):
        A, B, C = smooth_rank4_ktensor
        L = np.zeros((len(B), len(B)))
        for i, L_i in enumerate(L):
            if i-1 >= 0:
                L_i[i-1] = -1
                L_i[i] += 1
            if i+1 < len(B):
                L_i[i+1] = -1
                L_i[i] += 1
        return L

    def test_gradient_finite_difference(self, rank4_kruskal_tensor):
        X = rank4_kruskal_tensor.construct_tensor()
        cp_opt = self.CP(4, max_its=10000, convergence_tol=self.convergence_tol)
        cp_opt._init_fit(
            X=X, max_its=cp_opt.max_its, initial_decomposition=None, importance_weights=None
        )
        flattened = base.flatten_factors(cp_opt.decomposition)
        flattened_true = base.flatten_factors(rank4_kruskal_tensor)
        loss = cp_opt._flattened_loss
        derivative = cp_opt._flattened_gradient

        assert check_grad(loss, derivative, flattened) < 1e-5
        assert check_grad(loss, derivative, flattened_true) < 1e-5
    
    def check_decomposition(self, kruskal_tensor, *, test_closeness=True, test_fms=True, **additional_params):

        X = kruskal_tensor.construct_tensor()
        cp_opt = self.CP(4, max_its=10000, convergence_tol=self.convergence_tol, **additional_params)
        estimated_ktensor = cp_opt.fit_transform(X)
        estimated_ktensor.normalize_components()

        if test_closeness:
            assert 1 - np.linalg.norm(X - estimated_ktensor.construct_tensor())/np.linalg.norm(X) > 0.999
        if test_fms:
            assert metrics.factor_match_score(
                kruskal_tensor.factor_matrices, estimated_ktensor.factor_matrices
            )[0] > 1-1e-7
        
    def test_rank4_decomposition(self, rank4_kruskal_tensor):
        self.check_decomposition(rank4_kruskal_tensor)

    #def test_rank4_nonnegative_decomposition(self, nonnegative_rank4_kruskal_tensor):
    #    self.check_decomposition(nonnegative_rank4_kruskal_tensor, non_negativity_constraints=[True, True, True])

    def test_weighted_rank4_decomposition(self, rank4_kruskal_tensor):
        X = rank4_kruskal_tensor.construct_tensor()
        mask = np.random.uniform(0, 1, X.shape) > 0.5

        new_X = X.copy()
        new_X[~mask] = np.random.uniform(0, 1, (~mask).sum())

        cp_opt = self.CP(4, max_its=10000, convergence_tol=self.convergence_tol)
        estimated_ktensor = cp_opt.fit_transform(new_X, importance_weights=mask)

        assert 1 - np.linalg.norm(X - estimated_ktensor.construct_tensor())/np.linalg.norm(X) > 0.999
        assert metrics.factor_match_score(
            rank4_kruskal_tensor.factor_matrices, estimated_ktensor.factor_matrices
        )[0] > 1e-7


    def test_weighted_gradient_finite_difference(self, rank4_kruskal_tensor):
        X = rank4_kruskal_tensor.construct_tensor()
        mask = np.random.uniform(0, 1, X.shape) > 0.5

        new_X = X.copy()
        new_X[~mask] = np.random.uniform(0, 1, (~mask).sum())

        cp_opt = self.CP(4, max_its=10000, convergence_tol=self.convergence_tol)
        cp_opt._init_fit(
            X=new_X, max_its=cp_opt.max_its, initial_decomposition=None, importance_weights=mask
        )
        flattened = base.flatten_factors(cp_opt.decomposition)
        flattened_true = base.flatten_factors(rank4_kruskal_tensor)
        loss = cp_opt._flattened_loss
        derivative = cp_opt._flattened_gradient

        assert check_grad(loss, derivative, flattened) < 1e-5
        assert check_grad(loss, derivative, flattened_true) < 1e-5
    
    def test_smooth_gradient_finite_difference(self, smooth_rank4_ktensor, laplacian):
        factor_constraints = [{}, {'tikhonov_matrix': laplacian}, {}]

        X = smooth_rank4_ktensor.construct_tensor()
        cp_opt = self.CP(4, max_its=10000, convergence_tol=self.convergence_tol, factor_constraints=factor_constraints)
        cp_opt._init_fit(
            X=X, max_its=cp_opt.max_its, initial_decomposition=None, importance_weights=None
        )
        flattened = base.flatten_factors(cp_opt.decomposition)
        flattened_true = base.flatten_factors(smooth_rank4_ktensor)
        loss = cp_opt._flattened_loss
        derivative = cp_opt._flattened_gradient

        assert check_grad(loss, derivative, flattened) < 1e-5
        assert check_grad(loss, derivative, flattened_true) < 1e-5
    
    def test_smooth_loss_different_than_unregularised(self, smooth_rank4_ktensor, laplacian):
        factor_constraints = [{}, {'tikhonov_matrix': laplacian}, {}]

        X = smooth_rank4_ktensor.construct_tensor()
        cp_opt = self.CP(4, max_its=10000, convergence_tol=self.convergence_tol)
        cp_opt._init_fit(
            X=X, max_its=cp_opt.max_its, initial_decomposition=None, importance_weights=None
        )
        cp_smooth_opt = self.CP(4, max_its=10000, convergence_tol=self.convergence_tol, factor_constraints=factor_constraints)
        cp_smooth_opt._init_fit(
            X=X, max_its=cp_opt.max_its, initial_decomposition=None, importance_weights=None
        )

        assert cp_opt._compute_loss(smooth_rank4_ktensor) != cp_smooth_opt._compute_loss(smooth_rank4_ktensor)

    def test_smooth_gradient_different_than_unregularised(self, smooth_rank4_ktensor, laplacian):
        factor_constraints = [{}, {'tikhonov_matrix': laplacian}, {}]

        X = smooth_rank4_ktensor.construct_tensor()
        cp_opt = self.CP(4, max_its=10000, convergence_tol=self.convergence_tol)
        cp_opt._init_fit(
            X=X, max_its=cp_opt.max_its, initial_decomposition=None, importance_weights=None
        )
        cp_smooth_opt = self.CP(4, max_its=10000, convergence_tol=self.convergence_tol, factor_constraints=factor_constraints)
        cp_smooth_opt._init_fit(
            X=X, max_its=cp_opt.max_its, initial_decomposition=None, importance_weights=None
        )

        def b_grad(cp_opt, factors): return cp_opt._compute_gradient(factors)[1]

        assert (b_grad(cp_opt, smooth_rank4_ktensor) != b_grad(cp_smooth_opt, smooth_rank4_ktensor)).all()