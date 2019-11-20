import numpy as np

from tenkit import utils


def test_get_signs():
    X = np.identity(10)
    signs = 2*np.ones(10, dtype=int)*np.random.randint(0, 2, 10, dtype=int) - 1

    A = np.diag(signs)

    assert np.allclose(utils.get_signs(A, X)[0], signs)
