import numpy as np

try:
    from numba import jit, prange
except ImportError:
    withjit = False
    jsafe_range = range
else:
    withjit = True
    make_fast = jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    jsafe_range = prange


def kron_binary_vectors(u, v):
    n, = u.shape
    m, = v.shape
    kprod = u[:, np.newaxis]*v[np.newaxis, :]
    return kprod.reshape(n*m)


def khatri_rao_binary(A, B):
    """Calculates the Khatri-Rao product of A and B
    
    A and B have to have the same number of columns.
    """
    I, K = A.shape
    J, K = B.shape

    out = np.empty((I * J, K))
    # for k in range(K)
        # out[:, k] = kron_binary_vectors(A[:, k], B[:, k])
    for i, row in enumerate(A):
        out[i*J:(i+1)*J] = row[np.newaxis, :]*B
    #for i in jsafe_range(I):
    #    out[i*J:(i+1)*J] = A[i]*B
    return out

#if withjit:
#    khatri_rao_binary = make_fast(khatri_rao_binary)


def khatri_rao(*factors, skip=None):
    """Calculates the Khatri-Rao product of a list of matrices.
    
    Also known as the column-wise Kronecker product
    
    Parameters:
    -----------
    *factors: np.ndarray list
        List of factor matrices. The matrices have to all have 
        the same number of columns.
    skip: int or None (optional, default is None)
        Optional index to skip in the product. If None, no index
        is skipped.
        
    Returns:
    --------
    product: np.ndarray
        Khatri-Rao product. A matrix of shape (prod(N_i), M)
        Where prod(N_i) is the product of the number of rows in each
        matrix in `factors`. And M is the number of columns in all
        matrices in `factors`. 
    """
    factors = list(factors).copy()
    if skip is not None:
        factors.pop(skip)

    num_factors = len(factors)
    product = factors[0]

    for i in range(1, num_factors):
        product = khatri_rao_binary(product, factors[i])
    return product


def kron(*factors):
    factors = list(factors).copy()
    num_factors = len(factors)
    product = factors[0]

    for i in range(1, num_factors):
        product = kron_binary(product, factors[i])
    return product

def kron_binary(A, B):
    n, m = A.shape
    p, q = B.shape
    kprod = A[:, np.newaxis, :, np.newaxis]*B[np.newaxis, :, np.newaxis, :]
    return kprod.reshape(n*p, m*q)



def matrix_khatri_rao_product(X, factors, mode):
    assert len(X.shape) == len(factors)
    if len(factors) == 3:
        return _mttkrp3(X, factors, mode)

    return unfold(X, mode) @ khatri_rao(*tuple(factors), skip=mode)


def _mttkrp3(X, factors, mode):
    if mode == 0:
        return X.reshape(X.shape[0], -1) @ khatri_rao(*tuple(factors), skip=mode)
    elif mode == 1:
        return _mttkrp_mid(X, factors)
    elif mode == 2 or mode == -1:
        return np.moveaxis(X, -1, 0).reshape(X.shape[-1], -1) @ khatri_rao(
            *tuple(factors), skip=mode
        )


def _mttkrp_mid(tensor, matrices):
    krp = khatri_rao(*matrices, skip=1)
    return _mttkrp_mid_with_krp(tensor, krp)


def _mttkrp_mid_with_krp(tensor, krp):
    shape = tensor.shape

    block_size = shape[-1]
    num_rows = shape[-2]
    num_cols = krp.shape[-1]
    product = np.zeros((num_rows, num_cols))
    for i in range(shape[0]):
        idx = i % shape[0]
        product += tensor[idx] @ krp[i * block_size : (i + 1) * block_size]

    return product


def unfold(A, n):
    """Unfold tensor to matricizied form.
    
    Parameters:
    -----------
    A: np.ndarray
        Tensor to unfold.
    n: int
        Defines which mode to unfold along.
        
    Returns:
    --------
    M: np.ndarray
        The mode-n unfolding of `A`
    """

    M = np.moveaxis(A, n, 0).reshape(A.shape[n], -1)
    return M


def fold(M, n, shape):
    """Fold a matrix to a higher order tensor.
    
    The folding is structured to refold an mode-n unfolded 
    tensor back to its original form.
    
    Parameters:
    -----------
    M: np.ndarray
        Matrix that corresponds to a mode-n unfolding of a 
        higher order tensor
    n: int
        Mode of the unfolding
    shape: tuple or list
        Shape of the folded tensor
        
    Returns:
    --------
    np.ndarray
        Folded tensor of shape `shape`
    """
    newshape = list(shape)
    mode_dim = newshape.pop(n)
    newshape.insert(0, mode_dim)

    return np.moveaxis(np.reshape(M, newshape), 0, n)


def unflatten_factors(A, rank, sizes):
    n_modes = len(sizes)
    offset = 0

    factors = []
    for i, s in enumerate(sizes):
        stop = offset + (s * rank)
        matrix = A[offset:stop].reshape(s, rank)
        factors.append(matrix)
        offset = stop
    return factors


def flatten_factors(factors):
    sizes = [np.prod(factor.shape) for factor in factors]
    offsets = np.cumsum([0] + sizes)[:-1]
    flattened = np.empty(np.sum(sizes))
    for offset, size, factor in zip(offsets, sizes, factors):
        flattened[offset : offset + size] = factor.ravel()
    return flattened


def ktensor(*factors, weights=None):
    """Creates a tensor from Kruskal factors, 
    
    Parameters
    ----------
    *factors : np.ndarray list
        List of factor matrices. All factor matrices need to
        have the same number of columns. 
    weights: np.ndarray (Optional)
        Vector array of shape (1, rank) that contains the weights 
        for each component of the Kruskal composition.
        If None, each factor matrix is assaign a weight of one.
    """
    if weights is None:
        weights = np.ones_like(factors[0])

    if len(weights.shape) == 1:
        weights = weights[np.newaxis, ...]

    shape = [f.shape[0] for f in factors]
    tensor = (weights * factors[0]) @ khatri_rao(*factors[1:]).T

    return fold(tensor, 0, shape=shape)
