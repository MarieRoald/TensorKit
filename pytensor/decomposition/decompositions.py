import numpy as np
import h5py
from abc import ABC, abstractmethod, abstractclassmethod
from .. import base
from .. import metrics
from .. import utils


class BaseDecomposedTensor(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def construct_tensor(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    def store(self, filename):
        with h5py.File(filename, 'w') as h5:
            self.store_in_hdf5_group(h5)
    
    @abstractmethod
    def store_in_hdf5_group(self, group):
        pass
    
    def _prepare_hdf5_group(self, group):
        group.attrs['type'] = type(self).__name__
    
    @classmethod
    def from_file(cls, filename):
        with h5py.File(filename) as h5:
            return cls.load_from_hdf5_group(h5)
    
    @abstractclassmethod
    def load_from_hdf5_group(cls, group):
        pass
    
    @classmethod
    def _check_hdf5_group(cls, group):
        tensortype = None
        if 'type' in group.attrs:
            tensortype = group.attrs['type']

        if not group.attrs['type'] == cls.__name__:
            raise Warning(f'The `type` attribute of the HDF5 group is not'
                          f' "{cls.__name__}, but "{group.attrs["type"]}"\n.'
                          'This might mean that you\'re loading the wrong tensor file')


class KruskalTensor(BaseDecomposedTensor):
    fm_template = 'factor_matrix{:03d}'

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

        shape = [f.shape[0] for f in self.factor_matrices]
        tensor = (self.weights[np.newaxis] * self.factor_matrices[0]) @ base.khatri_rao(*self.factor_matrices[1:]).T

        return base.fold(tensor, 0, shape=shape)

    def reset_weights(self):
        self.weights *= 0
        self.weights += 1

    def normalize_components(self, update_weights=True, eps=1e-15):
        """Set all factor matrices to unit length. Updates the weights if `update_weights` is True.

        Arguments:
        ----------
        update_weights : bool
            If true, then the weights of this Kruskal tensor will be set to the product of the
            component norms.
        """
        for i, factor_matrix in enumerate(self.factor_matrices):
            norms = np.linalg.norm(factor_matrix, axis=0)
            self.factor_matrices[i][...] = factor_matrix/(norms[np.newaxis] + eps)
            if update_weights:
                self.weights *= norms
        
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

    def store_in_hdf5_group(self, group):
        self._prepare_hdf5_group(group)

        group.attrs['n_factor_matrices'] = len(self.factor_matrices)
        group.attrs['rank'] = self.rank

        for i, factor_matrix in enumerate(self.factor_matrices):
            group[self.fm_template.format(i)] = factor_matrix
        
        group['weights'] = self.weights
        
    @classmethod
    def load_from_hdf5_group(cls, group):
        cls._check_hdf5_group(group)

        factor_matrices = [
            group[cls.fm_template.format(i)][...]
                for i in range(group.attrs['n_factor_matrices'])
        ]
        weights = group['weights'][...]

        return cls(factor_matrices, weights)
    
    def __getitem__(self, item):
        return self.factor_matrices[item]

    def factor_match_score(self, decomposition, weight_penalty=True, fms_reduction='min'):
        assert decomposition.rank == self.rank

        return metrics.factor_match_score(self.factor_matrices, 
                                          decomposition.factor_matrices, 
                                          weight_penalty=weight_penalty, 
                                          fms_reduction=fms_reduction)
    
    def get_sign_scores(self, X):
        sign_scores = []
        for n, factor_matrix in enumerate(self.factor_matrices):
            sign_scores.append(utils.get_signs(factor_matrix, base.unfold(X, n))[1])
        
        return sign_scores
    
    def get_signs(self, X):
        sign_scores = self.get_sign_scores(X)
        signs = [np.ones(self.rank, dtype=int) for _ in self.factor_matrices]
        for rank in range(self.rank):
            single_rank_sign_scores = [
                single_mode_sign_scores[rank] for single_mode_sign_scores in sign_scores
            ]

            for factor_num, factor_sign_score in enumerate(single_rank_sign_scores):
                signs[factor_num][rank] = np.sign(factor_sign_score)

            # Find the mode that should not be flipped
            single_rank_signs = np.sign(single_rank_sign_scores)
            if np.prod(single_rank_signs) == -1:
                wrongly_flipped = np.argmin(np.abs(single_rank_sign_scores))
                signs[wrongly_flipped][rank] *= -1
        
        return signs
            


class EvolvingTensor(BaseDecomposedTensor):
    B_template = "B_{:03d}"
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
    def factor_matrices(self):
        return [self.A, self.B, self.C]
    
    @property
    def shape(self):
        """The shape of the tensor created by the `construct_tensor` function.
        """
        matrix_width = max([m.shape[0] for m in self.B])
        return [self.A.shape[0], matrix_width, self.C.shape[0]]
    
    def construct_slices(self):
        """Construct the data slices.
        """
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
        """Construct the datatensor from the factors. 
        Zero padding will be used if the tensor is irregular.
        """
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

    def store_in_hdf5_group(self, group):
        self._prepare_hdf5_group(group)

        group.attrs['rank'] = self.rank
        group.attrs['all_same_size'] = self.all_same_size
        group.attrs['warning'] = self.warning
        group.attrs['num_Bs'] = len(self.B)

        group['A'] = self.A
        for k, Bk in enumerate(self.B):
            group[self.B_template.format(k)] = Bk
        group['C'] = self.C
        
    @classmethod
    def load_from_hdf5_group(cls, group):
        cls._check_hdf5_group(group)

        A = group['A'][...]
        B = [group[cls.B_template.format(k)][...] for k in range(group.attrs['num_Bs'])]
        C = group['C'][...]
        warning = group.attrs['warning']
        all_same_size = group.attrs['all_same_size']

        return cls(A, B, C, all_same_size=all_same_size, warning=warning)
    
    def __getitem__(self, item):
        if item == 0:
            return self.A
        elif item == 1:
            return self.B
        elif item == 2:
            return self.C
        else:
            raise IndexError
        
        
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
    pm_template = 'projection_matrix_{:03d}'
    def __init__(self, A, blueprint_B, C, projection_matrices, warning=True):
        r"""A tensor whose second mode evolves over the third mode according to the PARAFAC2 constraints.

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
    def random_init(cls, sizes, rank, non_negativity=None):

        # TODO: Check if we should use rand or randn

        if isinstance(sizes[1], int):
            all_same_size = True
            sizes = list(sizes)
            sizes[1] = [sizes[1]]*sizes[2]
        else:
            all_same_size = False

        if non_negativity == None:
            non_negativity = [False, False, False]
            

        
        A = np.random.rand(sizes[0], rank)
        blueprint_B = np.identity(rank)
        C = np.random.rand(sizes[2], rank) + 0.1

        projection_matrices = []

        for second_mode_size in sizes[1]:
            q, r = np.linalg.qr(np.random.randn(second_mode_size, rank))
            projection_matrices.append(q[:, :rank])

        
        return cls(A, blueprint_B, C, projection_matrices, all_same_size)

    def store_in_hdf5_group(self, group):
        self._prepare_hdf5_group(group)

        group.attrs['rank'] = self.rank
        group.attrs['all_same_size'] = self.all_same_size
        group.attrs['warning'] = self.warning
        group.attrs['n_projection_matrices'] = len(self.projection_matrices)

        group['A'] = self.A
        group['blueprint_B'] = self.blueprint_B
        group['C'] = self.C
        for i, pm in enumerate(self.projection_matrices):
            group[self.pm_template.format(i)] = pm
        
    @classmethod
    def load_from_hdf5_group(cls, group):
        cls._check_hdf5_group(group)

        A = group['A'][...]
        blueprint_B = group['blueprint_B'][...]
        C = group['C'][...]
        warning = group.attrs['warning']

        projection_matrices = [
            group[cls.pm_template.format(i)][...]
                for i in range(group.attrs['n_projection_matrices'])
        ]

        return cls(A, blueprint_B, C, projection_matrices, warning=warning)

    def factor_match_score(self, decomposition, weight_penalty=True, fms_reduction='min'):
        assert decomposition.rank == self.rank

        factors1 = [self.A, np.array(self.B).reshape(-1, self.rank), self.C]
        factors2 = [decomposition.A, np.array(decomposition.B).reshape(-1, self.rank), decomposition.C]

        return metrics.factor_match_score(factors1, 
                                          factors2, 
                                          weight_penalty=weight_penalty, 
                                          fms_reduction=fms_reduction)
