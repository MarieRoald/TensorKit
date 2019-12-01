from abc import ABC, abstractclassmethod, abstractmethod

import h5py
import numpy as np
import scipy.linalg
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
from scipy.optimize import nnls


def rightsolve(A, B):
    """Solve the equation X*A = B wrt X.
    """
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    S[S != 0] = 1/S[S != 0]

    return B @ (Vh.T * S @ U.T)


def non_negative_rightsolve(A, B):
    """Solve the equation X*A = B wrt X under nonnegativity constraints.
    """
    # Discussion tracking in Enron Email Using PARAFAC has non negative updates
    if len(B.shape) == 1:
        B = B[np.newaxis, B]

    x = np.zeros((B.shape[0], A.shape[0]))
    for i, b_i in enumerate(B):
        x[i, :], _ = nnls(A.T, b_i) 

    return x


def orthogonal_rightsolve(A, B):
    """Solve the equation XA = B wrt X with orthogonality on X
    """
    return orthogonal_solve(A.T, B.T).T


def orthogonal_solve(A, B):
    """Solve the equation AX = B wrt X with orthogonality on X
    """
    U, S, Vh = np.linalg.svd(B.T@A, full_matrices=False)
    S_tol = max(U.shape) * S[0] * (1e-16)
    should_keep = (S > S_tol).astype(float)

    return (Vh.T * should_keep)@ U.T


def add_rightsolve_ridge(rightsolve, ridge_penalty):
    def ridge_rightsolve(A, B):
        n, m = A.shape
        p, q = B.shape
        A_ = np.concatenate(
            [A, np.sqrt(ridge_penalty)*np.identity(n)],
            axis=1
        )
        B_ = np.concatenate(
            [B, np.zeros((p, n))],
            axis=1
        )

        return rightsolve(A_, B_)
    return ridge_rightsolve


def add_rightsolve_coupling(rightsolve, coupled_factor_matrix, coupling_penalty):
    def coupling_rightsolve(A, B):
        p, q = A.shape
        A_ = np.concatenate(
            [A, np.sqrt(coupling_penalty)*np.identity(p)],
            axis=1
        )
        B_ = np.concatenate(
            [B, np.sqrt(coupling_penalty)*coupled_factor_matrix],
            axis=1
        )
        return rightsolve(A_, B_)
    return coupling_rightsolve


class NotConvergedError(Exception):
    pass


def create_tikhonov_rightsolve(tikhonov_matrix):
    def tikhonov_rightsolve(A, B):
        """Solve min ||XA - B||^2 + tr(X^T L X)

        Assume that A has low rank.
        """
        if not sparse.issparse(tikhonov_matrix):
            perturbed_data = B@A.T
            return scipy.linalg.solve_sylvester(tikhonov_matrix, A@A.T, perturbed_data)

        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        perturbed_data = B@(Vh.T*s)

        solution = np.empty((B.shape[0], A.shape[0]))

        identity = sparse.identity(tikhonov_matrix.shape[0])
        for r, s_r in enumerate(s):
            cg_solution = spla.cg(tikhonov_matrix + identity*s_r**2, perturbed_data[:, r], atol=0)
            if cg_solution[1] > 0:
                raise NotConvergedError
            solution[:, r] = cg_solution[0]

        return solution@U.T

    return tikhonov_rightsolve


def kron_binary_vectors(u, v):
    """Efficient Kronecker product between two vectors.
    """
    n, = u.shape
    m, = v.shape
    kprod = u[:, np.newaxis]*v[np.newaxis, :]
    return kprod.reshape(n*m)


# TODO: Test the speed of tensorly's KR computation. Maybe use their expression.
def khatri_rao_binary(A, B):
    """Calculates the Khatri-Rao product of A and B
    
    A and B have to have the same number of columns.
    """
    I, K = A.shape
    J, K = B.shape

    out = np.empty((I * J, K))
    # for k in range(K)
        # out[:, k] = kron_binary_vectors(A[:, k], B[:, k])
    # Equivalent but faster with C-contiguous arrays
    for i, row in enumerate(A):
        out[i*J:(i+1)*J] = row[np.newaxis, :]*B
    return out


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
    """Efficient Kronecker product of multiple matrices.
    """
    factors = list(factors).copy()
    num_factors = len(factors)
    product = factors[0]

    for i in range(1, num_factors):
        product = kron_binary(product, factors[i])
    return product


def kron_binary(A, B):
    """Efficient Kronecker product of the matrix A and B.
    """
    n, m = A.shape
    p, q = B.shape
    kprod = A[:, np.newaxis, :, np.newaxis]*B[np.newaxis, :, np.newaxis, :]
    return kprod.reshape(n*p, m*q)


def matrix_khatri_rao_product(X, factors, mode):
    """Compute the matricised tensor times Khatri Rao product along given mode.

    Parameters
    ----------
    X : np.ndarray
        Tensor
    factors : List[np.ndarray]
        List of factor matrices, the i-th factor matrix has shape [X.shape[i], rank]
    mode : int
        Which mode to unfold the tensor along. Should be between 0 and /len(factors) - 1)
    """
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


def unflatten_factors(flattened_factors, rank, sizes):
    """Transform a flattened set of factor matrices to a list of numpy arrays.

    Parameters
    ----------
    flattened_factors : np.ndarray
        One dimensional numpy array as returned by ``flatten_factors``.
    rank : int
        The rank of the decomposition
    sizes : Iterable[int]
        The length of each mode of the tensor the factor matrices represent.
    """
    n_modes = len(sizes)
    offset = 0

    factors = []
    for i, s in enumerate(sizes):
        stop = offset + (s * rank)
        matrix = flattened_factors[offset:stop].reshape(s, rank)
        factors.append(matrix)
        offset = stop
    return factors


def flatten_factors(factor_matrices):
    """Transform the list of factor matrices to a vector.

    Inverted by ``unflatten_factors``.
    """
    sizes = [np.prod(factor.shape) for factor in factor_matrices]
    offsets = np.cumsum([0] + sizes)[:-1]
    flattened = np.empty(np.sum(sizes))
    for offset, size, factor in zip(offsets, sizes, factor_matrices):
        flattened[offset : offset + size] = factor.ravel()
    return flattened
