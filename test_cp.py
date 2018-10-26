import numpy as np
from pytest import fixture, approx
import cp 
import base
from test_base import random_factors
from scipy import optimize


"""
TODO: (tests)
- Test normalize_factor and normalize_factors
- Test initialization
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

def test_scipy_check_weighted_grad():
    sizes = [5, 3, 10]
    rank = 2

    tensor = _random_tensor(sizes)
    factors = _random_factors(sizes, rank)
    flattened_factors = base.flatten_factors(factors)
    W = np.round(np.random.random(tensor.shape)).astype(np.int)

    args = (rank, sizes, tensor, W)
    check_grad_error = optimize.check_grad(cp._cp_weighted_loss_scipy, cp._cp_weighted_grad_scipy, flattened_factors, *args)
    assert check_grad_error==approx(0, abs=1e-5)


def test_weighted_loss_with_all_ones_same_as_regular_loss():
    sizes = [5, 3, 10]
    rank = 2

    factors = _random_factors(sizes, rank)
    tensor = base.ktensor(*tuple(factors))

    W = np.ones_like(tensor)
    assert cp.cp_loss(factors, tensor) == approx(cp.cp_weighted_loss(factors, tensor, W))

def test_weighted_grad_with_all_ones_same_as_regular_loss():
    sizes = [5, 3, 10]
    rank = 2

    factors = _random_factors(sizes, rank)
    tensor = base.ktensor(*tuple(factors))

    grads = cp.cp_grad(factors, tensor)

    W = np.ones_like(tensor)
    weighted_grads = cp.cp_weighted_grad(factors, tensor, W)

    for grad, weighted_grad in zip(grads, weighted_grads):
        assert np.allclose(grad, weighted_grad)

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


    

    



