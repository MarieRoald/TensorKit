import tempfile
from functools import partial

import numpy as np
import pytest
from scipy import sparse

from tenkit import base
from tenkit.decomposition.decompositions import *


class TestRightsolve:
    random = partial(np.random.uniform, 0, 1)
    def rightsolve(self, *args, **kwargs):
        return base.rightsolve(*args, **kwargs)

    @pytest.fixture
    def nonnegative_A(self):
        return np.array(
            [
                [2, 1, 0],
                [1, 0, 1]
            ]
        )
    
    @pytest.fixture
    def nonnegative_A_orthogonal_component(self):
        return np.array(
            [[1, -2, -1]]
        )

    def test_solvable_system_vector(self):
        A = self.random((2, 3))
        x = self.random((1, 2))
        b = x@A
        
        assert np.allclose(self.rightsolve(A, b), x)
    
    def test_solvable_system_matrix(self):
        A = self.random((2, 3))
        X = self.random((1, 2))
        B = X@A

        assert np.allclose(self.rightsolve(A, B), X)
    
    def test_least_squares_vector(self, nonnegative_A, nonnegative_A_orthogonal_component):
        x = np.array([[1, 1]])
        b = x@nonnegative_A + 0.5*nonnegative_A_orthogonal_component
        assert np.all(b >= 0), "If this fails, then the error lies in the test."

        assert np.allclose(self.rightsolve(nonnegative_A, b), x)
    
    def test_least_squares_matrix(self, nonnegative_A, nonnegative_A_orthogonal_component):
        X = np.array(
            [
                [1, 1],
                [1, 2]
            ]
        )
        B = X@nonnegative_A + 0.5*nonnegative_A_orthogonal_component
        assert np.all(B >= 0), "If this fails, then the error lies in the test."

        assert np.allclose(self.rightsolve(nonnegative_A, B), X)


class TestNonnegativeRightsolve(TestRightsolve):
    def rightsolve(self, *args, **kwargs):
        return base.non_negative_rightsolve(*args, **kwargs)
    
    def test_least_squares_matrix_negative_solution(self, nonnegative_A, nonnegative_A_orthogonal_component):
        X = np.array(
            [
                [2, -1],
                [2,  1]
            ]
        )
        Xn = X.copy()
        Xn[Xn < 0] = 0
        B = X@nonnegative_A - nonnegative_A_orthogonal_component

        assert np.all(B >= 0), "If this fails, then the error lies in the test."
        assert np.all(self.rightsolve(nonnegative_A, B) >= 0)


class TestOrthogonalRightsolve:
    def rightsolve(self, *args, **kwargs):
        return base.orthogonal_rightsolve(*args, **kwargs)

    @pytest.fixture
    def orthogonal_matrix(self):
        return np.linalg.qr(np.random.randn(100, 10))[0]
    
    @pytest.fixture
    def nonorthogonal_matrix(self):
        return np.random.randn(10, 50)
    
    def test_orthogonal_rightsolve(self, orthogonal_matrix, nonorthogonal_matrix):
        X = orthogonal_matrix
        Y = nonorthogonal_matrix
        product = X@Y
        assert np.linalg.norm(X - self.rightsolve(Y, product))/np.linalg.norm(X) < 1e-5


def test_add_ridge_same_as_soft_coupling():
    A = np.random.randn(2, 3)
    x = np.random.randn(5, 2)
    b = x@A

    ridge_rightsolve = base.add_rightsolve_ridge(base.rightsolve, 1)
    coupled_rightsolve = base.add_rightsolve_coupling(base.rightsolve, np.zeros((5, 2)), 1)

    assert np.allclose(ridge_rightsolve(A, b), coupled_rightsolve(A, b))


def test_tikhonov_rightsolve():
    rank = 4
    shape = (50, 60)
    for k in range(50):
        A = np.random.standard_normal((shape[0], rank))*(2**(k%20 - 10))
        B = np.random.standard_normal((shape[1], rank))
        X = A@B.T

        L = np.zeros([shape[1]]*2)
        for i in range(60):
            L[i, i-1] = -1
            L[i, i] = 2
            L[i, (i+1)%shape[1]] = -1

        rightsolve = base.create_tikhonov_rightsolve(L)
        B_approx = rightsolve(A.T, X.T)

        gradient = (B_approx@A.T - X.T)@A + L@B_approx

        assert np.linalg.norm(gradient.ravel(), np.inf) < 1e-4, str(k)

        L = sparse.csr_matrix(L)

        rightsolve = base.create_tikhonov_rightsolve(L)
        B_approx = rightsolve(A.T, X.T)

        gradient = (B_approx@A.T - X.T)@A + L@B_approx

        assert np.linalg.norm(gradient.ravel(), np.inf) < 1e-1, f"{k}, {k%20 - 10}"
