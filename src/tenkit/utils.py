from copy import deepcopy

import numpy as np

from . import base


def flip_factors(factor_matrices):
    factor_matrices = deepcopy(factor_matrices)
    signs = []
    for i, factor in enumerate(factor_matrices):
        sign = np.sign(np.mean(np.sign(factor), axis=0))

        # Resolve zero-signs so they are equal to sign of first nonzero element
        for k, s in enumerate(sign):
            if s == 0:
                sign[k] = _find_first_nonzero_sign(factor[:, k])

        factor_matrices[i] *= sign
        signs.append(sign)
    return factor_matrices, signs


def get_pca_loadings(Y, rank):
    """Returns the pca loadings of the Y matrix.
    """
    U, S, Vh = np.linalg.svd(Y)
    A = (Vh.T @ np.diag(S))[:, :rank]
    return A




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

    tensor += noise_factor * noise
    return tensor, factors, norms, noise


def create_random_uniform_factors(sizes, rank):
    factors = [np.random.uniform(size=(size, rank)) for size in sizes]
    factors, norms = normalize_factors(factors)
    return factors, norms


def create_non_negative_data(sizes, rank, noise_factor=0):
    factors, norms = create_random_uniform_factors(sizes=sizes, rank=rank)
    tensor = base.ktensor(*tuple(factors))

    noise = np.random.randn(*sizes)
    noise /= np.linalg.norm(noise)
    noise *= np.linalg.norm(tensor)

    tensor += noise_factor * noise
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
    return factor / (norms + eps), norms


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


def get_signs(factor_matrix, X):
    """Find the correct signs of the factor matrix.
    If data matrix is not passed, then the sign is set so the factor mean is positive.
    If the data matrix is passed, then the sign is set so the factors are positively
    correlated with the data matrix.

    Arguments
    ---------
    factor_matrix : np.ndarray
        Factor matrix with unknown sign
    X : np.ndarray
        Data matrix

    Returns
    -------
    sign : int
    sign_weight : float
    """
    if X is None:
        sign_weight = factor_matrix.sum(axis=0)
        sign_weight[sign_weight == 0] = 1
        return np.sign(sign_weight), sign_weight

    X_described_by_factors = np.linalg.lstsq(factor_matrix, X)[0]
    sign_weight = np.sum(
        np.sign(X_described_by_factors) * X_described_by_factors**2,
        axis=1
    )

    return np.sign(sign_weight), sign_weight
    
def signfix_evolving_factors(data_tensor, evolving_factor, evolve_over_factor, data_driven=True):
    fixed_evolving_factor = evolving_factor.copy()
    fixed_evolve_over_factor = evolve_over_factor.copy()
    for k in range(len(evolving_factor)):
        signs = get_signs(data_tensor[k], evolving_factor[k], data_driven=data_driven)[0]
        
        fixed_evolving_factor[k] *= signs
        fixed_evolve_over_factor[k] *= signs
    
    return fixed_evolving_factor, fixed_evolve_over_factor

def signfix_normal_factors(data_tensor, fixing_factors, flipping_factors, data_driven=True):
    unfolded_data_tensor = data_tensor.reshape(data_tensor.shape[0], -1)
    signs = get_signs(unfolded_data_tensor, fixing_factors, data_driven)[0]
    
    return fixing_factors*signs[np.newaxis], flipping_factors*signs[np.newaxis]

def fix_signs_evolving_tensor(evolving_tensor, data_tensor):
    evolving_factor = evolving_tensor.B
    evolve_over_factor = evolving_tensor.C

    fixed_evolving_factor, fixed_evolve_over_factor = signfix_evolving_factors(data_tensor, evolving_factor, evolve_over_factor)
    
    fixed_evolving_tensor = deepcopy(evolving_tensor)

    fixed_evolving_tensor._B = fixed_evolving_factor
    fixed_evolving_tensor._C = fixed_evolve_over_factor
    return fixed_evolving_tensor

def iter_checkpoints(h5_checkpoint_file):
    for groupname in sorted(h5_checkpoint_file):
        if groupname.startswith('checkpoint'):
            yield h5_checkpoint_file[groupname]


def slice_SSE(X_slices1, X_slices2):
    SSE = 0
    for slice1, slice2, in zip(X_slices1, X_slices2):
        SSE += np.sum((slice1-slice2)**2)
    return SSE
