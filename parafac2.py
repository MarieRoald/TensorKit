import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pprint import pprint
import itertools

import cp
import base

def get_loadings(Y, rank):
    U, S, Vh = np.linalg.svd(Y)
    A =(Vh.T@np.diag(S))[:,:rank]
    return A

def init_A(X, rank):
    
    K = len(X)
    Y = np.zeros([X[0].shape[1], X[0].shape[1]])
    
    for k in range(K):
        Y += X[k].T@X[k]

    A = get_loadings(Y, rank)

    return A

def init_F(X, rank):
    return np.identity(rank)

def init_D_k(X, rank):
    K = len(X)
    D = np.zeros((rank, rank, K))
    
    for k in range(K):
        D[..., k] = np.identity(rank)
    return D
        
def initialization(X, rank):
    A = init_A(X, rank)
    F = init_F(X, rank)
    D_k = init_D_k(X, rank)
    return A, F, D_k
    

def update_P_k(X, A, F, D_k, rank):
    K = len(X)
    
    P_k = []
    for k in range(K):
        (A.T @ X[k].T).shape
        U, S, Vh = np.linalg.svd(F @ D_k[...,k] @ (A.T @ X[k].T),  full_matrices=False)
        P_k.append(Vh.T@U.T)
    return P_k

def update_F_A_D(X,P_k, F, A, D, rank):
    C = np.diagonal(D)
    K = len(X)
    J = X[0].shape[1]
    factors = [F, A, C]
    weights = np.ones((3, rank))
    X_hat = np.empty((rank, J, K))
    for k in range(K):
        X_hat[...,k] = P_k[k].T @ X[k]
        
    factors, weights = cp.update_als_factors(X_hat, factors, weights)
    F, A, C = factors
    
    for k in range(K):
        D[...,k] = np.diag(C[k])
    
    return F, A, D


def compose_from_factors(P_k, F, A, D_k):
    K = len(P_k)
    
    X_pred = []
    for k, P in enumerate(P_k):
        D = D_k[...,k]
        F_k = P @ F
        
        X_k = F_k @ D @ A.T
        X_pred.append(X_k)
        
    return X_pred

def loss(X, X_pred):
    error = 0
    for x, x_pred in zip(X, X_pred):
        error += np.linalg.norm(x-x_pred)**2
    return error

def _check_convergence(iteration, X, pred, f_prev, verbose):
    #TODO: better name and docstring
    #MSE = np.mean((pred - X)**2)
    #mse = MSE(X_list, pred)
    f = loss(X,pred)
    

    REL_FUNCTION_ERROR = (f_prev-f)/f_prev

    if verbose:
        print(f'{iteration}: f is {f:4f}, improvement is {REL_FUNCTION_ERROR:4f}')
    return REL_FUNCTION_ERROR,f

def parafac2_als(X, rank, max_its=1000, convergence_th=1e-10, verbose=True):
    """Compute parafac2 decomposition with alternating least squares."""


    A, F, D_k = initialization(X, rank)
    P_k = update_P_k(X,A,F,D_k, rank)
    pred = compose_from_factors(P_k, F, A, D_k)
    f_prev = loss(X, pred)
    REL_FUNCTION_CHANGE = np.inf

    for it in range(max_its):
        if abs(REL_FUNCTION_CHANGE) < convergence_th:
            break

        P_k = update_P_k(X,A,F,D_k, rank)
        F, A, D_k = update_F_A_D(X,P_k, F, A, D_k, rank)
        pred = compose_from_factors(P_k, F, A, D_k)
        
        REL_FUNCTION_CHANGE, f_prev = _check_convergence(it, X, pred, f_prev, verbose)
    return P_k, F, A, D_k
    