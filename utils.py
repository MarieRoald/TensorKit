import numpy as np
import base
import cp

def create_random_factors(sizes, rank):
    factors = [np.random.randn(size, rank) for size in sizes]
    factors, norms = normalize_factors(factors)
    return factors, norms

def create_data(sizes, rank, noise_factor=0):
    factors, norms = create_random_factors(sizes=sizes, rank=rank)
    tensor = base.ktensor(*tuple(factors))

    noise = np.random.randn(*sizes)
    noise /= np.linalg.norm(noise)
    noise *= np.linalg.norm(tensor)

    tensor += noise_factor*noise
    return tensor, factors, norms, noise

def create_random_uniform_factors(sizes, rank, ):
    factors = [np.random.uniform(size=(size, rank)) for size in sizes]
    factors, norms = normalize_factors(factors)
    return factors, norms

def create_non_negative_data(sizes, rank, noise_factor=0):
    factors, norms = create_random_uniform_factors(sizes=sizes, rank=rank)
    tensor = base.ktensor(*tuple(factors))

    noise = np.random.randn(*sizes)
    noise /= np.linalg.norm(noise)
    noise *= np.linalg.norm(tensor)

    tensor += noise_factor*noise
    return tensor, factors, norms, noise

def permute_factors(permutation, factors):
    return [factor[:, permutation] for factor in factors]

def permute_factors_and_weights(permutation, factors, weights):
    permuted_factors = [factor[:, permutation] for factor in factors]
    permuted_weights = weights[list(permutation)]
    return permuted_factors, permuted_weights

def normalize_factor(factor, eps=1e-15):
    """Normalizes the columns of a factor matrix. 
    
    Parameters:
    -----------
    factor: np.ndarray
        Factor matrix to normalize.
    eps: float
        Epsilon used to prevent division by zero.

    Returns:
    --------
    np.ndarray:
        Matrix where the columns are normalized to length one.
    np.ndarray:
        Norms of the columns before normalization.
    """
    norms = np.linalg.norm(factor, axis=0, keepdims=True)
    return factor/(norms+eps), norms

def normalize_factors(factors):
    """Normalizes the columns of each element in list of factors
    
    Parameters:
    -----------
    factor: list of np.ndarray
        List containing factor matrices to normalize.

    Returns:
    --------
    list of np.ndarray:
        List containing matrices where the columns are normalized 
        to length one.
    list of np.ndarray:
        List containing the norms of the columns from before 
        normalization.
    """
    normalized_factors = []
    norms = []

    for factor in factors:
        normalized_factor, norm = normalize_factor(factor)
        normalized_factors.append(normalized_factor)
        norms.append(norm)

    return normalized_factors, norms

def _find_first_nonzero_sign(factor):
    """Returns the sign of the first nonzero element of `factor`
    """
    sign = 0
    for el in factor:
        if sign != 0:
            break
        sign = np.sign(el)
    
    return sign
def prepare_for_comparison(factors):
    """Normalize factors and flip the signs.

    This normalization makes it easier to compare the results.
    TODO: more details.
    """
    normalized_factors, norms = normalize_factors(factors)
    signs = []
    for i, factor in enumerate(normalized_factors):
        sign = np.sign(np.mean(np.sign(factor), axis=0))

        # Resolve zero-signs so they are equal to sign of first nonzero element
        for k, s in enumerate(sign):
            if s == 0:
                sign[k] = _find_first_nonzero_sign(factor[:, k])
        
        normalized_factors[i] *= sign
        signs.append(sign)
    return normalized_factors, signs, norms