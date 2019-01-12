import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import base
import utils
from log import Logger

from scipy import optimize
import itertools


"""
TODO:
- Nonnegative alternative to svd init
- Random init within bounds
- Function to generate random mask that takes amount of missing data as input
- Maybe svdinit of cp_wopt should imputate missing values with mean?
- 
"""

def _initialize_factors_random(shape, rank):
    """Random initialization of factor matrices"""
    factors = [np.random.randn(s, rank) for s in shape]
    return factors

def _initialize_factors_svd(X, rank):
    """SVD based initialization of factor matrices"""
    n_modes = len(X.shape)
    factors =[]
    if rank > min(X.shape):
        raise ValueError(f'SVD initialisation does not work when rank is larger than the smallest dimension of X.\
                          (rank:{rank}, dimensions: {X.shape})')
    for i in range(n_modes):
        u, s, vh = np.linalg.svd(base.unfold(X,i))

        factors.append(u[:,:rank])
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
    
    return [utils.normalize_factor(f)[0] for f in factors], weights
    
def _compute_V(factors, skip_mode):
    """Compute left hand side of least squares problem."""
    rank = factors[0].shape[1]
    
    V = np.ones((rank, rank))
    for i, factor in enumerate(factors):
        if i == skip_mode:
            continue
        V *= factor.T@factor
    return V

def compute_V(factors, skip_mode):
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
    #rhs = (base.unfold(X, mode) @ base.khatri_rao(*tuple(factors), skip=mode)).T
    rhs = base.mttkrp(X, factors, mode).T
    new_factor,res, rank, s = np.linalg.lstsq(V.T, rhs)
    new_factor = new_factor.T
    
    return utils.normalize_factor(new_factor)
    
def update_als_factors(X, factors, weights):
    """Updates factors with alternating least squares."""
    num_axes = len(X.shape)
    for axis in range(num_axes):
        factors[axis], weights[axis] = update_als_factor(X, factors, axis)
        
    return factors, weights

def _check_convergence(iteration, X, pred, f_prev, verbose):
    #TODO: better name and docstring
    MSE = np.mean((pred - X)**2)
    f = np.linalg.norm(X-pred)

    REL_FUNCTION_ERROR = (f_prev-f)/f_prev

    if verbose:
        print(f'{iteration}: The MSE is {MSE:4f}, f is {f:4f}, improvement is {REL_FUNCTION_ERROR:4f}')
    return REL_FUNCTION_ERROR,f
    
def cp_als(X, rank, max_its=1000, convergence_th=1e-10, init='random', verbose=True):
    """Compute cp decomposition with alternating least squares."""
    
    X_norm = np.sqrt(np.sum(X**2))
    num_axes = len(X.shape)
    
    factors, weights = initialize_factors(X, rank, method=init)
    
    pred = base.ktensor(*tuple(factors), weights=weights.prod(axis=0))
    f_prev = np.linalg.norm(X-pred)
    REL_FUNCTION_CHANGE = np.inf
    
    for it in range(max_its):
        if abs(REL_FUNCTION_CHANGE) < convergence_th:
            break
            
        factors, weights = update_als_factors(X, factors, weights)
    
        pred = base.ktensor(*tuple(factors), weights=weights.prod(axis=0))
        REL_FUNCTION_CHANGE, f_prev = _check_convergence(it, X, pred, f_prev, verbose)
        
    return factors, weights.prod(axis=0)


def cp_loss(factors, tensor):
    """Loss function for a CP (CANDECOMP/PARAFAC) model.

    Loss(X) = ||X - [[F_0, F_1, ..., F_k]]||^2

    Parameters:
    -----------
    factor: list of np.ndarray
        List containing factor matrices for a CP model.
    tensor: np.ndarray
        Tensor we are attempting to model with CP.

    Returns:
    --------
    float:
        Loss value
    """
    reconstructed_tensor = base.ktensor(*factors)
    return 0.5*np.linalg.norm(tensor-reconstructed_tensor)**2

def cp_weighted_loss(factors, tensor, W):
    weighted_tensor = W*tensor

    reconstructed_tensor = base.ktensor(*factors)
    weighted_reconstructed_tensor = W*reconstructed_tensor

    return 0.5*np.linalg.norm(weighted_tensor-weighted_reconstructed_tensor)**2

def cp_weighted_grad(factors, X, W):
    Y = W*X

    reconstructed_tensor = base.ktensor(*factors)
    Z = W*reconstructed_tensor

    grads = []
    for mode in range(len(factors)):
        grads.append((base.unfold(Z, n=mode) - base.unfold(Y, n=mode)) 
                      @ base.khatri_rao(*tuple(factors), skip=mode))
    return grads

def _cp_weighted_grad_scipy(A, *args):    
    rank, sizes, X, W = args
    n_modes = len(sizes)

    factors = base.unflatten_factors(A, rank, sizes)
    grad = cp_weighted_grad(factors, X, W)
    return base.flatten_factors(grad)

def _cp_weighted_loss_scipy(A, *args):
    rank, sizes, X, W = args
    factors = base.unflatten_factors(A, rank, sizes)
    return cp_weighted_loss(factors, X, W)

def cp_grad(factors, X):
    """Gradients for CP loss."""
    grad_A = []
    for mode in range(len(factors)):
        grad_A.append(- base.matrix_khatri_rao_product(X, factors, mode)
                      + factors[mode] @ compute_V(factors, mode))
    return grad_A

def _cp_loss_scipy(A, *args):
    """The CP loss for scipy.optimize.minimize

    from scipy documentation:
    scipy.optimize.minimize takes an objective function fun(x, *args)
    where x is an 1-D array with shape (n,) and args is a tuple of the 
    fixed parameters needed to completely specify the function.

    Parameters:
    -----------
    A: np.ndarray 
        1D array of shape (n,) containing the parameters to minimize over.
    *args: tuple
        The fixed parameters needed to completely specify the function.
    """
    rank, sizes, X = args
    factors = base.unflatten_factors(A, rank, sizes)
    return cp_loss(factors, X)



def _cp_grad_scipy(A, *args):
    """

    """
    rank, sizes, X = args
    n_modes = len(sizes)

    factors = base.unflatten_factors(A, rank, sizes)
    grad = cp_grad(factors, X)
    return base.flatten_factors(grad)

def cp_opt(X, rank, method='cg', max_its=1000, lower_bounds=None, upper_bounds=None, gtol=1e-10, init='random'):
    sizes = X.shape
    options = {'maxiter': max_its, 'gtol': gtol}


    args = (rank, sizes, X)

    logger = Logger(args=args, loss=_cp_loss_scipy, grad=_cp_grad_scipy)

    initial_factors, _ = initialize_factors(X, rank, method=init)
    initial_factors_flattened = base.flatten_factors(initial_factors)

    bounds = create_bounds(lower_bounds, upper_bounds, sizes, rank)

    result = optimize.minimize(fun=_cp_loss_scipy, method=method, x0=initial_factors_flattened, 
                               jac=_cp_grad_scipy, bounds=bounds, args=args, options=options, callback=logger.log)

    factors = base.unflatten_factors(result.x, rank, sizes)
    return factors, result, initial_factors, logger

def create_bounds(lower_bounds, upper_bounds, sizes, rank):

    if (lower_bounds is None) and (upper_bounds is None):
        return None

    if lower_bounds is None:
        lower_bounds = -np.inf
    if upper_bounds is None:
        upper_bounds = np.inf

    lower_bounds = _create_bounds(lower_bounds, sizes, rank)
    upper_bounds = _create_bounds(upper_bounds, sizes, rank)

    return _bounds_scipy(lower_bounds, upper_bounds)


def _create_bounds(bounds, sizes, rank):

    if _isiterable(bounds) == False:
        if bounds is None:
            bounds = np.inf
        bounds = [bounds]*len(sizes)
    # TODO: assert bounds length = sizes length?
    full_bounds = []

    for size, bound in zip(sizes, bounds):
        full_bounds.append(np.full((size, rank), fill_value=bound))

    return full_bounds

def _bounds_scipy(lower_bounds, upper_bounds):
    upper = base.flatten_factors(upper_bounds)
    lower = base.flatten_factors(lower_bounds)

    #return list(zip(lower,upper)) 
    return optimize.Bounds(lb=lower, ub=upper)
def _isiterable(var):
    try:
        iter(var)
    except TypeError:
        return False
    else:
        return True


def cp_wopt(X, W, rank, method='cg', max_its=1000, gtol=1e-10, init='random'):
    sizes = X.shape
    options = {'maxiter': max_its, 'gtol': gtol}

    args = (rank, sizes, X, W)

    initial_factors, _ = initialize_factors(X, rank, method=init)
    initial_factors_flattened = base.flatten_factors(initial_factors)

    logger = Logger(args=args, loss=_cp_weighted_loss_scipy, grad=_cp_weighted_grad_scipy)

    result = optimize.minimize(fun=_cp_weighted_loss_scipy, method=method, x0=initial_factors_flattened, 
                               jac=_cp_weighted_grad_scipy, args=args, options=options, callback=logger.log)    

    factors = base.unflatten_factors(result.x, rank, sizes)

    return factors, result, initial_factors, logger

if __name__ == "__main__":
    X = loadmat('datasets/aminoacids.mat')['X'][0][0]['data']
    X = X/np.linalg.norm(X)
    factors, weights = cp_als(X, 3, convergence_th=1e-5)

    fig, axes = plt.subplots(3,1, figsize=(18,12))
    for i in range(3):
        axes[i].plot(factors[i])
    plt.show()

    """
    data = loadmat('datasets/toydata.mat')
    A = np.array(data["A"])
    B = np.array(data["B"])
    X1 = A@(np.array([[1,0],[0,1]])@B.T)
    X2 = A@(np.array([[3,0],[0,2]])@B.T)
    XX = np.stack([X1,X2],2)

    factors, weights = cp_als(XX, 2, convergence_th=1e-15)

    fig, axes  = plt.subplots(2,2,figsize=(14,7))
    axes[0,0].plot(A)
    axes[0,1].plot(B)

    axes[0,0].set_title("A")
    axes[0,1].set_title("B")
    axes[0,0].legend(["$a_1$", "$a_2$"])
    axes[0,1].legend(["$b_1$", "$b_2$"])

    axes[1,0].plot(np.array([[1,-1]])*factors[0]/np.linalg.norm(factors[0], axis=0, keepdims=True))
    axes[1,1].plot(np.array([[-1,1]])*factors[1]/np.linalg.norm(factors[1], axis=0, keepdims=True)) 

    axes[1,0].set_title("Recovered A")
    axes[1,1].set_title("Recovered B")
    axes[1,0].legend(["$a_1$", "$a_2$"])
    axes[1,1].legend(["$b_1$", "$b_2$"])
    plt.show()
    """