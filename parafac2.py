import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pprint import pprint
import itertools

import cp
import base
import utils


def _get_pca_loadings(Y, rank):
    """Returns the pca loadings of the Y matrix.
    """
    U, S, Vh = np.linalg.svd(Y)
    A = (Vh.T@np.diag(S))[:, :rank]
    return A

def _init_A(X, rank, init_scheme='svd'):
    """A is initialised as the PCA loadings from the sum of covariance matrices.
    """
    K = len(X)
    J = X[0].shape[1]
    if init_scheme == 'svd':
        Y = np.zeros([J, J])
        
        for k in range(K):
            Y += X[k].T@X[k]

        A = _get_pca_loadings(Y, rank)
    else:
        A = np.random.rand(J,rank)
    return A

def _init_F(X, rank):
    """F is initialised as an identity matrix.
    """
    return np.identity(rank)

def _init_D_k(X, rank, init_scheme='svd'):
    """D is initialised as a sequence of identity matrices.
    """
    K = len(X)
    C = _init_C(X, rank, init_scheme=init_scheme)

    di = np.diag_indices(rank)
    D = np.zeros((rank, rank, K))

    for k in range(K):
        D[...,k][di] = C[k]

    return D

def _init_C(X, rank, init_scheme='svd'):
    K = len(X)
    if init_scheme=='svd':
        C = np.ones((K, rank))
    else:
        C = np.random.rand(K, rank)
    return C
        
def _init_parafac2(X, rank, init_scheme='svd'):

    if init_scheme!='svd' and init_scheme!='random':
        raise ValueError(f'"init_scheme" has to be "random" or "svd". \
                        {init_scheme} is not a valid initialisation option. ')

    A = _init_A(X, rank, init_scheme=init_scheme)
    F = _init_F(X, rank)
    D_k = _init_D_k(X, rank, init_scheme=init_scheme)
    P_k = _update_P_k(X, F, A, D_k, rank)
    return P_k, A, F, D_k

def _update_P_k(X, F, A, D_k, rank):
    K = len(X)
    
    P_k = []
    for k in range(K):
        U, S, Vh = np.linalg.svd(F @ D_k[...,k] @ (A.T @ X[k].T),  full_matrices=False)
        
        S_tol = max(U.shape)*S[0]*(1e-16)
        should_keep = np.diag(S > S_tol).astype(float)

        P_k.append(Vh.T @ should_keep @ U.T)   # Should_keep = diag([1, 1, ..., 1, 0, 0, ..., 0]) -> the zeros correspond to small singular values
                                               # Following Rasmus Bro's PARAFAC2 MATLAB script, which sets P_k = Q_k(Q_k'Q_k)^(-0.5) (line 524)
                                               #      Where the power is done by truncating very small singular values (for numerical stability)
    return P_k

def _update_F_A_D(X, P_k, F, A, D_k, rank):
    # TODO: try without normalizing the weights
    C = np.diagonal(D_k)
    K = len(X)
    J = X[0].shape[1]
    factors = [F, A, C]
    weights = np.ones((3, rank))
    X_hat = np.empty((rank, J, K))

    for k in range(K):
        X_hat[...,k] = P_k[k].T @ X[k]
        
    # MATLAB performs up to five PARAFAC updates
    factors, weights = cp.update_als_factors(X_hat, factors, weights)
    F, A, C = factors[0], factors[1], factors[2]
    weights = weights.prod(0, keepdims=True)
    F *= weights
    # weights = weights.prod(0, keepdims=True)**(1/3)
    # F, A, C = (weights*factor for factor in factors)
    
    for k in range(K):
        D_k[...,k] = np.diag(C[k])
    
    return F, A, D_k

def _update_parafac2(X, P_k, F, A, D_k, rank):
    P_k = _update_P_k(X, F, A, D_k, rank)
    F, A, D_k = _update_F_A_D(X, P_k, F, A, D_k, rank)
    return P_k, F, A, D_k


def update_als_factor_p2(X, factors, mode):
    """Solve least squares problem to get factor for one mode."""
    V = cp._compute_V(factors, mode)
    
    # Solve least squares problem
    #rhs = (base.unfold(X, mode) @ base.khatri_rao(*tuple(factors), skip=mode)).T
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
    K = len(P_k)
    
    X_pred = []
    for k, P in enumerate(P_k):
        D = D_k[...,k]
        F_k = P @ F
        
        X_k = F_k @ D @ A.T
        X_pred.append(X_k)
        
    return X_pred

def _parafac2_loss(X, X_pred):
    error = 0
    for x, x_pred in zip(X, X_pred):
        error += np.linalg.norm(x-x_pred)**2
    return error

def _MSE(X, X_pred):
    X_ = np.concatenate([x.ravel() for x in X])
    X_pred_ = np.concatenate([xp.ravel() for xp in X_pred])
    return np.mean((X_ - X_pred_)**2)

def _check_convergence(iteration, X, pred, prev_loss, verbose):
    loss = _parafac2_loss(X, pred)
    
    REL_FUNCTION_ERROR = (prev_loss-loss)/prev_loss

    if verbose:
        print(f'{iteration:4d}: loss is {loss:4.2f}, improvement is {REL_FUNCTION_ERROR:4.2f}')
    return REL_FUNCTION_ERROR, loss

def parafac2_als(X, rank, max_its=1000, convergence_th=1e-10, verbose=True, init_scheme='svd'):
    """Compute parafac2 decomposition with alternating least squares."""
    P_k, A, F, D_k = _init_parafac2(X, rank, init_scheme=init_scheme)

    pred = compose_from_parafac2_factors(P_k, F, A, D_k)
    prev_loss = _parafac2_loss(X, pred)
    REL_FUNCTION_CHANGE = np.inf

    for it in range(max_its):
        if abs(REL_FUNCTION_CHANGE) < convergence_th:
            break

        P_k, F, A, D_k = _update_parafac2(X, P_k, F, A, D_k, rank)
        pred = compose_from_parafac2_factors(P_k, F, A, D_k)

        
        REL_FUNCTION_CHANGE, prev_loss = _check_convergence(it, X, pred, prev_loss,
                                                            verbose)
    return P_k, F, A, D_k
    
def create_random_orthogonal_factor(dimension, rank):
    return np.linalg.qr(np.random.randn(dimension, rank))[0]

def create_parafac2_components(n_values, m, rank):
    sizes = (rank, m, len(n_values))

     
    (F, A, C), _ = utils.create_random_factors(sizes, rank)
    P_k = [create_random_orthogonal_factor(ni, rank) for ni in n_values]

    D_k = np.zeros((rank, rank, len(n_values)))
    for k in range(len(n_values)):
        D_k[..., k] = np.diag(C[k])
    return P_k, F, A, D_k

    