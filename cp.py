import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import base

def normalize_factor(factor):
    """Normalizes the columns of a factor matrix. 
    
    Parameters:
    -----------
    factor: np.ndarray
        Factor matrix to normalize

    Returns:
    --------
    np.ndarray:
        Matrix where the columns are normalized to length one.
    np.ndarray:
        Norms of the columns before normalization.
    """
    norms = np.linalg.norm(factor, axis=0, keepdims=True)
    return factor/norms, norms

def _initialize_factors_random(shape, rank):
    """Random initialization of factor matrices"""
    factors = [np.random.randn(s, rank) for s in shape]
    return factors

def _initialize_factors_svd(X, rank):
    """SVD based initialization of factor matrices"""
    n_modes = len(X.shape)
    factors =[]
    for i in range(n_modes):
        u, s, vh = np.linalg.svd(base.unfold(X,i))
        factors.append(u[:rank].T)
    return factors
    
def initialize_factors(X, rank, method='random'):
    if method == 'random':
        factors = _initialize_factors_random(X.shape, rank)
    elif method == 'svd':
        factors = _initialize_factors_svd(X, rank)
    else:
        #TODO: ERROR OR SOMETHING
        factors = _initialize_factors_random(X.shape, rank)
    
    weights = np.ones((len(X.shape), rank))
    
    return [normalize_factor(f)[0] for f in factors], weights
    
def _compute_V(factors, skip_mode):
    """Compute left hand side of least squares problem."""
    rank = factors[0].shape[1]
    
    V = np.ones((rank, rank))
    for i, factor in enumerate(factors):
        if i == skip_mode:
            continue
        V *= factor.T@factor
    return V

def update_als_factor(X, factors, mode):
    """Solve least squares problem to get factor for one mode."""
    V = _compute_V(factors, mode)
    
    # Solve least squares problem
    rhs = (base.unfold(X, mode) @ base.khatri_rao(*tuple(factors), skip=mode)).T
    new_factor = np.linalg.solve(V.T, rhs).T
    
    return normalize_factor(new_factor)
    
def update_als_factors(X, factors, weights):
    """Updates factors with alternating least squares."""
    num_axes = len(X.shape)
    for axis in range(num_axes):
        factors[axis], weights[axis] = update_als_factor(X, factors, axis)
        
    return factors, weights

def _check_convergence(iteration, X, pred, f_prev, verbose):
    #TODO: better name and docstring and everything
    MSE = np.mean((pred - X)**2)
    f = np.linalg.norm(X-pred)

    REL_FUNCTION_ERROR = (f_prev-f)/f_prev

    if verbose:
        print(f'{iteration}: The MSE is {MSE:4f}, f is {f:4f}, improvement is {REL_FUNCTION_ERROR:4f}')
    return REL_FUNCTION_ERROR,f
    
def cp_decomposition(X, rank, num_its=1000, convergence_th=1e-10, verbose=True):
    """Compute cp decomposition with alternating least squares."""
    
    X_norm = np.sqrt(np.sum(X**2))
    num_axes = len(X.shape)
    
    factors, weights = initialize_factors(X, rank, method='svd')
    
    pred = base.ktensor(*tuple(factors), weights=weights.prod(axis=0))
    f_prev = np.linalg.norm(X-pred)
    REL_FUNCTION_CHANGE = np.inf
    
    for it in range(num_its):
        if abs(REL_FUNCTION_CHANGE) < convergence_th:
            break
            
        factors, weights = update_als_factors(X, factors, weights)
    
        pred = base.ktensor(*tuple(factors), weights=weights.prod(axis=0))
        REL_FUNCTION_CHANGE, f_prev = _check_convergence(it, X, pred, f_prev, verbose)
        
    
    return factors, weights.prod(axis=0)


if __name__ == "__main__":
    X = loadmat('datasets/aminoacids.mat')['X'][0][0]['data']
    X = X/np.linalg.norm(X)
    factors, weights = cp_decomposition(X, 3, convergence_th=1e-5)

    fig, axes = plt.subplots(3,1, figsize=(18,12))
    for i in range(3):
        axes[i].plot(factors[i])
    plt.show()