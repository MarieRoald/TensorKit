import numpy as np
from pytest import fixture, approx
import cp 
import base
from test_base import random_factors
from scipy import optimize


"""
TODO: (tests)
- Test that the loss is zero for exact decomposition
- Test that the gradient of loss is zero for exact decomposition
- 
"""


def _random_factors(sizes, rank):
    return [np.random.random((size, rank)) for size in sizes]

def _random_tensor(sizes):
    return np.random.random(sizes)

def test_scipy_check_grad():
    sizes = [5, 3, 10]
    rank = 2

    tensor = _random_tensor(sizes)
    factors = _random_factors(sizes, rank)
    flattened_factors = base.flatten_factors(factors)

    args = (rank, sizes, tensor)
    check_grad_error = optimize.check_grad(cp._cp_loss_scipy, cp._cp_grad_scipy, flattened_factors, *args)
    assert check_grad_error==approx(0, abs=1e-5)



