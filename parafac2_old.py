import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pprint import pprint
import itertools

import cp
import base
import utils

def parafac2_checkpoint_saver(h5_group, it, parafac2_factors):
    if 'checkpoint' not in h5_group:
        h5_group.create_group('checkpoint')

    chk_group = h5_group('checkpoint')
    assert f'it_{it:05d}' not in chk_group

    chk_group = chk_group.create_group(f'it_{it:05d}')

    chk_group.attrs['num_modes'] = 3

    P_k, F, A, D_k = parafac2_factors

    chk_group[f'factor_F'] = F
    chk_group[f'factor_A'] = A
    chk_group[f'factor_D_k'] = D_k

    for k,P in enumerate(P_k):
        chk_group[f'factor_P_{k}'] = P

def parafac2_checkpoint_loader(h5_group, it):
    chk_group = h5_group('checkpoint')
    chk_group = chk_group[f'it_{it:05d}']

    D_k = chk_group[f'factor_D_k']
    K = D_k.shape[2]

    F = chk_group[f'factor_F']
    A = chk_group[f'factor_A']

    P_k = []
    for k in range(K):
        P = chk_group[f'factor_P_{k}'][...]
        P_k.append(P)

    return P_k, F, A, D_k

def _get_pca_loadings(Y, rank):
    """Returns the pca loadings of the Y matrix.
    """
    U, S, Vh = np.linalg.svd(Y)
    A = (Vh.T @ np.diag(S))[:, :rank]
    return A


def _init_A(X, rank, init_scheme="svd"):
    """A is initialised as the PCA loadings from the sum of covariance matrices.
    """
    K = len(X)
    J = X[0].shape[1]
    if init_scheme == "svd":
        Y = np.zeros([J, J])

        for k in range(K):
            Y += X[k].T @ X[k]

        A = _get_pca_loadings(Y, rank)
    else:
        A = np.random.rand(J, rank)
    return A


def _init_F(X, rank):
    """F is initialised as an identity matrix.
    """
    return np.identity(rank)


def _init_D_k(X, rank, init_scheme="svd"):
    """D is initialised as a sequence of identity matrices.
    """
    K = len(X)
    C = _init_C(X, rank, init_scheme=init_scheme)

    di = np.diag_indices(rank)
    D = np.zeros((rank, rank, K))

    for k in range(K):
        D[..., k][di] = C[k]

    return D


def _init_C(X, rank, init_scheme="svd"):
    K = len(X)
    if init_scheme == "svd":
        C = np.ones((K, rank))
    else:
        C = np.random.rand(K, rank)
    return C


def _init_parafac2(X, rank, init_scheme="svd"):

    if init_scheme != "svd" and init_scheme != "random":
        raise ValueError(
            f'"init_scheme" has to be "random" or "svd". \
                        {init_scheme} is not a valid initialisation option. '
        )

    A = _init_A(X, rank, init_scheme=init_scheme)
    F = _init_F(X, rank)
    D_k = _init_D_k(X, rank, init_scheme=init_scheme)
    P_k = _update_P_k(X, F, A, D_k, rank)
    return P_k, A, F, D_k


def _update_P_k(X, F, A, D_k, rank):
    K = len(X)

    P_k = []
    for k in range(K):
        U, S, Vh = np.linalg.svd(F @ D_k[..., k] @ (A.T @ X[k].T), full_matrices=False)

        S_tol = max(U.shape) * S[0] * (1e-16)
        should_keep = np.diag(S > S_tol).astype(float)

        P_k.append(
            Vh.T @ should_keep @ U.T
        )  # Should_keep = diag([1, 1, ..., 1, 0, 0, ..., 0]) -> the zeros correspond to small singular values
        # Following Rasmus Bro's PARAFAC2 MATLAB script, which sets P_k = Q_k(Q_k'Q_k)^(-0.5) (line 524)
        #      Where the power is done by truncating very small singular values (for numerical stability)
    return P_k


def _update_F_A_D(X, P_k, F, A, D_k, rank, non_negativity_constraints=None):
    # TODO: try without normalizing the weights
    C = np.diagonal(D_k)
    K = len(X)
    J = X[0].shape[1]
    factors = [F, A, C]
    weights = np.ones((1, rank))
    X_hat = np.empty((rank, J, K))

    for k in range(K):
        X_hat[..., k] = P_k[k].T @ X[k]

    # MATLAB performs up to five PARAFAC updates
    factors, weights = cp.update_als_factors(X_hat, factors, weights, non_negativity_constraints)
    F, A, C = factors[0], factors[1], factors[2]
    F *= weights
    # weights = weights.prod(0, keepdims=True)**(1/3)
    # F, A, C = (weights*factor for factor in factors)

    for k in range(K):
        D_k[..., k] = np.diag(C[k])

    return F, A, D_k


def _update_parafac2(X, P_k, F, A, D_k, rank, non_negativity_constraints):
    P_k = _update_P_k(X, F, A, D_k, rank)
    F, A, D_k = _update_F_A_D(X, P_k, F, A, D_k, rank, non_negativity_constraints)

    return P_k, F, A, D_k


def update_als_factor_p2(X, factors, mode):
    """Solve least squares problem to get factor for one mode."""
    V = cp._compute_V(factors, mode)

    # Solve least squares problem
    # rhs = (base.unfold(X, mode) @ base.khatri_rao(*tuple(factors), skip=mode)).T
    rhs = base.matrix_khatri_rao_product(X, factors, mode)
    new_factor = np.linalg.solve(V.T, rhs).T

    return new_factor


def update_als_factors_p2(X, factors):
    """Updates factors with alternating least squares."""
    num_axes = len(X.shape)
    for axis in range(num_axes):
        factors[axis] = update_als_factor_p2(X, factors, axis)

    return factors


def compose_from_parafac2_factors(P_k, F, A, D_k):
    if F is None:
        F = np.eye(P_k.shape[-1])

    K = len(P_k)

    X_pred = []
    for k, P in enumerate(P_k):
        D = D_k[..., k]
        F_k = P @ F

        X_k = F_k @ D @ A.T
        X_pred.append(X_k)

    return X_pred


def _parafac2_loss(X, X_pred):
    error = 0
    for x, x_pred in zip(X, X_pred):
        error += np.linalg.norm(x - x_pred) ** 2
    return error


def _MSE(X, X_pred):
    X_ = np.concatenate([x.ravel() for x in X])
    X_pred_ = np.concatenate([xp.ravel() for xp in X_pred])
    return np.mean((X_ - X_pred_) ** 2)

def _SSE(X, X_pred):
    X_ = np.concatenate([x.ravel() for x in X])
    X_pred_ = np.concatenate([xp.ravel() for xp in X_pred])
    return np.sum((X_ - X_pred_) ** 2) 


def _check_convergence(iteration, X, pred, prev_loss, verbose):
    loss = _parafac2_loss(X, pred)

    REL_FUNCTION_ERROR = (prev_loss - loss) / prev_loss

    if verbose:
        print(
            f"{iteration:4d}: loss is {loss:e}, improvement is {REL_FUNCTION_ERROR:4.2g}"
        )
    return REL_FUNCTION_ERROR, loss


def parafac2_als(
    X, rank, max_its=1000, convergence_th=1e-10, verbose=True, 
    non_negativity_constraints=None, init_scheme="svd", logger=None
):
    """Compute parafac2 decomposition with alternating least squares."""
    P_k, A, F, D_k = _init_parafac2(X, rank, init_scheme=init_scheme)

    pred = compose_from_parafac2_factors(P_k, F, A, D_k)
    prev_loss = _parafac2_loss(X, pred)
    REL_FUNCTION_CHANGE = np.inf

    for it in range(max_its):
        if abs(REL_FUNCTION_CHANGE) < convergence_th:
            break

        P_k, F, A, D_k = _update_parafac2(X, P_k, F, A, D_k, rank, non_negativity_constraints)
        pred = compose_from_parafac2_factors(P_k, F, A, D_k)

        REL_FUNCTION_CHANGE, prev_loss = _check_convergence(
            it, X, pred, prev_loss, verbose
        )
        if logger is not None:
            logger.log([P_k, F, A, D_k, pred])
    return P_k, F, A, D_k


def create_random_orthogonal_factor(dimension, rank):
    return np.linalg.qr(np.random.randn(dimension, rank))[0]


def create_parafac2_components(n_values, m, rank, C_offset=0.1, cross_product_value=0.4):
    P_k = [create_random_orthogonal_factor(ni, rank) for ni in n_values]
    F = create_random_F_factor(rank, cross_product_value=cross_product_value)

    A = create_random_A_factor(m, rank)

    C = create_random_C_factor(len(n_values), rank, offset=C_offset)
    D_k = get_D_k_from_C_factor(C)


    return P_k, F, A, D_k

def create_nn_parafac2_components(n_values, m, rank, C_offset=0.1, cross_product_value=0.4):
    P_k = [create_random_orthogonal_factor(ni, rank) for ni in n_values]
    F = create_random_F_factor(rank, cross_product_value=cross_product_value)

    A = create_random_A_factor(m, rank, non_negative=True)

    C = create_random_C_factor(len(n_values), rank, offset=C_offset)
    D_k = get_D_k_from_C_factor(C)


    return P_k, F, A, D_k


def create_random_F_factor(rank, cross_product_value=0.4):
    # From Kiers paper
    phi = np.ones(shape=(rank, rank))*cross_product_value
    np.fill_diagonal(phi, val=1)

    F = np.linalg.cholesky(phi)
    return F

def create_random_A_factor(m, rank, non_negative=False):
    if non_negative:
        A = np.random.uniform(size=(m,rank))
    else:
        A = np.random.randn(m, rank)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    return A

def create_random_C_factor(k, rank, offset=0.1):
    # From Kiers paper
    C = np.random.uniform(low=offset, high=1. + offset, size=(k, rank))
    return C

def get_D_k_from_C_factor(C):
    num_slices, rank = C.shape
    D_k = np.zeros((rank, rank, num_slices))
    for k in range(num_slices):
        D_k[..., k] = np.diag(C[k])
    return D_k


def get_F_k_from_P_k_and_F(P_k, F):
    F_k = [P @ F for P in P_k]
    return F_k

def check_parafac2_uniqueness(P_k, F, A, D_k, verbose=True):
    K = len(P_k)
    rank = F.shape[0]
    unique = True
    # K >= 4
    if K < 4:
        if verbose:
            print(f'K = {K} is < 4')
        unique = False
    # F^T T is positive definite
    if np.linalg.det(F) == 0:
        if verbose:
            print(f'det(F) = 0')
        unique = False
    if np.all(np.isreal(F)) is False:
        if verbose:
            print(f'F is not real')
        unique = False
    # A has full column rank
    if np.linalg.matrix_rank(A) != rank:
        if verbose:
            print(f'A does not have full column rank')
        unique = False
    # More than 4 singular D_k matrices
    C = np.diagonal(D_k)
    if np.count_nonzero(C.prod(axis=1)) > 4:
        if verbose:
            print(f'More than 4 singular D_k matrices')
        unique = False
    return unique
