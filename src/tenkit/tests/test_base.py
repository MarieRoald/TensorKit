import tempfile
import pytest
import numpy as np
from tenkit import base
from functools import partial


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