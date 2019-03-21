import numpy as np
from abc import ABC, abstractmethod
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
    # Equivalent but faster with C-contiguous arrays
    for i, row in enumerate(A):
        out[i*J:(i+1)*J] = row[np.newaxis, :]*B
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
        If None, each factor matrix is assign a weight of one.
    """
    if weights is None:
        weights = np.ones_like(factors[0])

    if len(weights.shape) == 1:
        weights = weights[np.newaxis, ...]

    shape = [f.shape[0] for f in factors]
    tensor = (weights * factors[0]) @ khatri_rao(*factors[1:]).T

    return fold(tensor, 0, shape=shape)


class BaseDecomposedTensor(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def construct_tensor(self):
        pass


# TODO: Size should probably not be property for performance reasons? Should check this.
class KruskalTensor(BaseDecomposedTensor):
    def __init__(self, factor_matrices, weights=None):
        self.rank = factor_matrices[0].shape[1]

        for i, factor_matrix in enumerate(factor_matrices):
            if factor_matrix.shape[1] != self.rank:
                raise ValueError(
                    f'All factor matrices must have the same number of columns. \n'
                    f'The first factor matrix has {self.rank} columns, whereas the {i}-th '
                    f'has {factor_matrix.shape[1]} columns.'
                )

        self.factor_matrices = factor_matrices
        if weights is None:
            weights = np.ones(self.rank)
        else:
            if len(weights) != self.rank:
                raise ValueError(
                    f'There must be as many weights as there are columns in the factor matrices.'
                    f'The factor matrices has {self.rank} columns, but there are {len(weights)} weights.'
                )
        self.weights = weights
    
    @property
    def shape(self):
        return [fm.shape[0] for fm in self.factor_matrices]
    
    def construct_tensor(self):
        if len(self.weights.shape) == 1:
            self.weights = self.weights[np.newaxis, ...]

        shape = [f.shape[0] for f in self.factor_matrices]
        tensor = (self.weights * self.factor_matrices[0]) @ khatri_rao(*self.factor_matrices[1:]).T

        return fold(tensor, 0, shape=shape)

    def normalize_components(self, update_weights=True):
        """Set all factor matrices to unit length. Updates the weights if `update_weights` is True.

        Arguments:
        ----------
        update_weights : bool
            If true, then the weights of this Kruskal tensor will be set to the product of the
            component norms.
        """
        weights = np.ones(self.rank)
        for i, factor_matrix in enumerate(self.factor_matrices):
            norms = np.linalg.norm(factor_matrix, axis=0)
            self.factor_matrices[i][...] = factor_matrix/norms[np.newaxis]
            weights *= norms
        
        if update_weights:
            self.weights[...] = weights
        
        return self

    @classmethod
    def random_init(cls, sizes, rank, random_method='normal'):
        """Construct a random Kruskal tensor with unit vectors as components and unit weights.

        Arguments:
        ----------
        sizes : tuple[int]
            The length of each mode of the generated Kruskal tensor. 
        rank : int
            Rank of the generated Kruskal tensor.
        random_method : str
            Which distribution to draw numbers from 'normal' or 'uniform'. All vectors are scaled to unit norm.
            If 'normal', a standard normal distribution is used. If 'uniform' a uniform [0, 1) distribution is used.
        """
        if random_method.lower() =='normal':
            factor_matrices = [np.random.randn(size, rank) for size in sizes]
        elif random_method.lower() =='uniform':
            factor_matrices = [np.random.uniform(size=(size, rank)) for size in sizes]
        else:
            raise ValueError("`random_method` must be either 'normal' or 'uniform'")
        
        return cls(factor_matrices).normalize_components(update_weights=False)




class EvolvingTensor(BaseDecomposedTensor):
    def __init__(self, A, B, C, all_same_size=True, warning=True):
        """A tensor whose second mode evolves over the third mode.

        Arguments:
        ----------
        factor_matrices : list
            List of factor matrices, the `evolve_mode`-th factor should
            either be a third order tensor or a list of matrices.
        all_same_size : Bool (default=True)
            Whether or not the constructed data is a tensor or a list of
            matrices with different sizes.
        warning : Bool (default=True)
            Whether or nor a warning should be raised when construct
            tensor is called if all the matrices are not the same size.
        """
        self.rank = A.shape[1]
        #self.factor_matrices = factor_matrices
        self._A = A
        self._B = B
        self._C = C

        self.warning = warning
        self.all_same_size = self.check_all_same_size(B)
        self.slice_shapes = [(self.A.shape[0], B_k.shape[0]) for B_k in self.B]
        self.num_elements = sum((shape[1] for shape in self.slice_shapes))

    def check_all_same_size(self, matrices):
        size = matrices[0].shape[0]
        for matrix in matrices:
            if size != matrix.shape[0]:
                return False
        return True

    @property
    def A(self):
        return self._A
    
    @property
    def B(self):
        return self._B
        
    @property
    def C(self):
        return self._C
    
    @property
    def shape(self):
        """The shape of the tensor created by the `construct_tensor` function.
        """
        matrix_width = max([m.shape[0] for m in self.B])
        return [self.A.shape[0], matrix_width, self.C.shape[0]]
    
    def construct_slices(self):
        slices = [None]*len(self.B)
        for k, matrix_size in enumerate(self.slice_shapes):
            slices[k] = self.construct_slice(k)
        
        return slices
            
    def construct_slice(self, k):
        """Construct the k-th slice along the third mode of the tensor.
        """
        loadings = self.A
        scores = self.C[k]*self.B[k]

        return loadings @ scores.T

    def construct_tensor(self):
        if self.warning and not self.all_same_size:
            raise Warning(
                'The factors have irregular shapes, zero padding will be used to construct tensor.\n'
                'Consider whether or not you want to call `construct_slices` instead.\n'
                'To supress this warning, pass warn=False at init.'
            )

        shape = self.shape
        constructed = np.zeros(shape)
        for k, _ in enumerate(self.slice_shapes):
            slice_ = self.construct_slice(k)
            constructed[:, :slice_.shape[1], k] = slice_
        return constructed
        
class ProjectedFactor:
    def __init__(self, factor, projection_matrices):
        self.factor = factor
        self.projection_matrices = projection_matrices
    
    def __getitem__(self, k):
        return self.projection_matrices[k]@self.factor
    
    def __len__(self):
        return len(self.projection_matrices)
    
    def as_list(self):
        return list(self)


class Parafac2Tensor(EvolvingTensor):
    def __init__(self, A, blueprint_B, C, projection_matrices, warning=True):
        """A tensor whose second mode evolves over the third mode according to the PARAFAC2 constraints.

        Let $X_k$ be the $k$-th slice of the matrix along the third mode. The tensor can then be
        described in the following manner
        $$X_k = A diag(C_k) B_k^T,$$
        where A is the factor matrix of the first mode, B_k is the k-th factor matrix of the second mode
        and C is the factor matrix of the third mode.

        The PARAFAC2 constraint is the following:
        $$B_k^T B_k = \Phi,$$
        for all $k$. Thus, B_k can be written as
        $$B_k = P_k B,$$
        with $P_k^TP_k = I$. 

        We call the $P_k$ matrices projection matrices and the $B$ matrix the blueprint matrix.
        Arguments:
        ----------
        factor_matrices : list[np.ndarray]
            A list of factor matrices, the second element should be the blueprint matrix.
        projection_matrices : list[np.ndarray]
        """
        self.rank = A.shape[1]
        self._A = A
        self._blueprint_B = blueprint_B
        self._C = C
        self._projection_matrices = tuple(projection_matrices)
        self._B = ProjectedFactor(blueprint_B, self._projection_matrices)


        self.all_same_size = self.check_all_same_size(projection_matrices)
        self.slice_shapes = [(self.A.shape[0], B_k.shape[0]) for B_k in self.B] 
        self.num_elements = sum(shape[1] for shape in self.slice_shapes)

        self.warning = warning


    @property
    def A(self):
        return self._A
        
    @property
    def C(self):
        return self._C
    
    @property
    def blueprint_B(self):
        return self._blueprint_B

    @property
    def projection_matrices(self):
        return self._projection_matrices
                
    @property
    def D(self):
        return np.array([np.diag(self.C[:, r]) for r in range(self.rank)])

    @classmethod
    def random_init(cls, sizes, rank):

        # TODO: Check if we should use rand or randn

        if isinstance(sizes[1], int):
            all_same_size = True
            sizes = list(sizes)
            sizes[1] = [sizes[1]]*sizes[2]
        else:
            all_same_size = False
            

        A = np.random.rand(sizes[0], rank)
        blueprint_B = np.identity(rank)
        C = np.random.rand(sizes[2], rank)

        projection_matrices = []

        for second_mode_size in sizes[1]:
            q, r = np.linalg.qr(np.random.randn(second_mode_size, rank))
            projection_matrices.append(q[:, :rank])

        
        return cls(A, blueprint_B, C, projection_matrices, all_same_size)
