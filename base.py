import numpy as np
def khatri_rao_binary(A,B):
    """Calculates the Khatri-Rao product of A and B
    
    A and B have to have the same number of columns.
    """
    
    I,K = A.shape
    J,K = B.shape
    
    out = np.empty(shape=[I*J,K])
    for k in range(K):
        out[:,k] = np.kron(A[:,k], B[:,k])
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
    
    factors = list(factors)
    if skip is not None:
        factors.pop(skip)
    
    num_factors = len(factors)
    product = factors[0]
    
    for i in range(1, num_factors):
        product = khatri_rao_binary(product, factors[i])
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
    
    M = np.moveaxis(A, n, 0).reshape(A.shape[n],-1)
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
    tensor = (weights*factors[0]) @ khatri_rao(*factors[1:]).T

    return fold(tensor, 0, shape=shape)