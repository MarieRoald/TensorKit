import numpy as np
from pytest import fixture, approx
import base 

np.random.seed(0)

"""
TODO: (tests)
- Test that the functions do not change the input
- 
"""

#---------------- Fixtures ----------------#
@fixture
def random_tensor():
    return np.random.random((2, 2, 2))

@fixture
def random_matrix():
    return np.random.random((2, 2))
    
@fixture
def random_factors():
    rank = 2
    sizes = [5, 3, 10]
    factors = []
    for size in sizes:
        factors.append(np.random.random((size, rank)))

    return factors

@fixture
def random_flattened_factors(random_factors):
    return base.flatten_factors(random_factors)
    
#----------------- Tests ------------------#
def test_unfold_inverted_by_fold(random_tensor):
    for i in range(len(random_tensor.shape)):
        assert np.array_equal(random_tensor, base.fold(base.unfold(random_tensor, i), i, random_tensor.shape))

def test_flatten_inverted_by_unflatten(random_factors):
    rank = random_factors[0].shape[1]
    sizes = [factor.shape[0] for factor in random_factors]
    unflattened_factors = base.unflatten_factors(base.flatten_factors(random_factors), rank, sizes)
    
    for random_factor, flattened_factor in zip(random_factors, unflattened_factors):
        assert np.array_equal(random_factor, flattened_factor)

def test_khatri_rao_associative(random_factors):
    A, B, C = random_factors
    prod1 = base.khatri_rao_binary(base.khatri_rao_binary(A, B), C)
    prod2 = base.khatri_rao_binary(A, base.khatri_rao_binary(B, C))
    assert np.allclose(prod1,prod2, rtol=1.e-10)