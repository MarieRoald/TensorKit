import numpy as np
from pytest import fixture, approx
import cp 
import base
from test_base import random_factors
from scipy import optimize


"""
TODO: (tests)
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


def test_loss_is_zero_for_exact_decomposition():
    sizes = [5, 3, 10]
    rank = 2

    factors = _random_factors(sizes, rank)
    tensor = base.ktensor(*tuple(factors))

    assert cp.cp_loss(factors, tensor) == approx(0)

def test_grad_is_zero_for_exact_decomposition():
    sizes = [5, 3, 10]
    rank = 2

    factors = _random_factors(sizes, rank)
    tensor = base.ktensor(*tuple(factors))

    grads = cp.cp_grad(factors, tensor)
    for grad in grads:
        assert np.all(grad<1e-10)


    

    



