import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pprint import pprint
import itertools

import cp
import base


def _get_pca_loadings(Y, rank):
    """Returns the pca loadings of the Y matrix.
    """
    U, S, Vh = np.linalg.svd(Y)
    A = (Vh.T@np.diag(S))[:, :rank]
    return A

def _init_A(X, rank):
    """A is initialised as the PCA loadings from the sum of covariance matrices.
    """
    K = len(X)
    Y = np.zeros([X[0].shape[1], X[0].shape[1]])
    
    for k in range(K):
        Y += X[k].T@X[k]

    A = _get_pca_loadings(Y, rank)

    return A

def _init_F(X, rank):
    """F is initialised as an identity matrix.
    """
    return np.identity(rank)

def _init_D_k(X, rank):
    """D is initialised as a sequence of identity matrices.
    """
    K = len(X)
    D = np.zeros((rank, rank, K))
    
    for k in range(K):
        D[..., k] = np.identity(rank)
    return D
        
def _init_parafac2(X, rank):
    A = _init_A(X, rank)
    F = _init_F(X, rank)
    D_k = _init_D_k(X, rank)
    P_k = _update_P_k(X, F, A, D_k, rank)
    return P_k, A, F, D_k

def _update_P_k(X, F, A, D_k, rank):
    K = len(X)
    
    P_k = []
    for k in range(K):
        U, S, Vh = np.linalg.svd(F @ D_k[...,k] @ (A.T @ X[k].T),  full_matrices=False)
        P_k.append(Vh.T@U.T)
    return P_k

def _update_F_A_D(X, P_k, F, A, D_k, rank):
    C = np.diagonal(D_k)
    K = len(X)
    J = X[0].shape[1]
    factors = [F, A, C]
    weights = np.ones((3, rank))
    X_hat = np.empty((rank, J, K))
    for k in range(K):
        X_hat[...,k] = P_k[k].T @ X[k]
        
    factors, weights = cp.update_als_factors(X_hat, factors, weights)

    weights = weights.prod(0, keepdims=True)**(1/3)
    F, A, C = (weights*factor for factor in factors)
    
    for k in range(K):
        D_k[...,k] = np.diag(C[k])
    
    return F, A, D_k

def _update_parafac2(X, P_k, F, A, D_k, rank):
    P_k = _update_P_k(X, F, A, D_k, rank)
    F, A, D_k = _update_F_A_D(X, P_k, F, A, D_k, rank)
    return P_k, F, A, D_k

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

def parafac2_als(X, rank, max_its=1000, convergence_th=1e-10, verbose=True):
    """Compute parafac2 decomposition with alternating least squares."""
    P_k, A, F, D_k = _init_parafac2(X, rank)

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
    