import numpy as np
import h5py
from abc import ABC, abstractmethod, abstractclassmethod
from .. import base
from .. import metrics


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


class CoupledTensors(BaseDecomposedTensor):
    def __init__(self, tensor_factors, matrices_factors, coupling_modes, weights=None, mat_weights=None):
        # tensor: the tensor-factors to be coupled, matrices: nested list of matrix-factors to couple, coupling_modes: list of modes
        if len(matrices_factors) != len(coupling_modes):
            raise ValueError('Coupled matrices was {0} but coupling modes was {1}'.format(len(matrices_factors), len(coupling_modes)))
        self.rank = tensor_factors[0].shape[1]
        #self.matrices_factors = matrices_factors
        self.coupling_modes = coupling_modes
        self._create_kruskals(tensor_factors, matrices_factors, weights=weights, mat_weights=mat_weights)
    
    @property
    def factor_matrices(self):
        return self.tensor.factor_matrices

    @property
    def uncoupled_factor_matrices(self):
        return [mat.factor_matrices[1] for mat in self.matrices]

    @property
    def coupled_factor_matrices(self):
        return [self.factor_matrices[i] for i in self.coupling_modes]
    
    def update_coupled_matrices(self):
        for i, mat in enumerate(self.matrices):
            mode = self.coupling_modes[i]
            mat.factor_matrices[0][...] = self.tensor.factor_matrices[mode]

    def _create_kruskals(self, tensor_factors, matrices_factors, weights=None, mat_weights=None):
        weights = np.ones(self.rank) if weights is None else weights
        mat_weights = [np.ones(self.rank) for _ in range(len(self.coupling_modes))] if mat_weights is None else mat_weights
        self.tensor = KruskalTensor(tensor_factors, weights=weights)
        self.matrices = [KruskalTensor([np.copy(tensor_factors[self.coupling_modes[i]]), mat],
                    weights=mat_weights[i]) for i, mat in enumerate(matrices_factors)]

    def construct_tensor(self):
        return self.tensor.construct_tensor()

    def construct_matrices(self):
        return [mat.construct_tensor() for mat in self.matrices]

    def reset_weights(self):
        for obj in [self.tensor] + self.matrices:
            obj.reset_weights()

    def normalize_components(self, update_weights=True, eps=1e-15):
        """Set all factor matrices to unit length. Updates the weights if `update_weights` is True.

        Arguments:
        ----------
        update_weights : bool
            If true, then the weights of this Kruskal tensor will be set to the product of the
            component norms.
        """
        self.update_coupled_matrices()
        for obj in [self.tensor] + self.matrices:
            obj.normalize_components(update_weights=update_weights)
        return self

    @classmethod
    def random_init(cls, tensor_sizes, rank, matrices_sizes, coupling_modes, random_method='normal'):
        """Construct a random Kruskal tensor coupled with n matrices, all with unit vectors as components and unit weights.
        """
       # check that modes are correct
        for i, size in enumerate(matrices_sizes):
            if size[0] != tensor_sizes[coupling_modes[i]]:
                print(size, tensor_sizes[coupling_modes[i]])
                raise ValueError('The coupling is not right.')

        if random_method.lower() == 'normal':
            matrices_factors = [np.random.randn(size[1], rank) for size in matrices_sizes]
            tensor_factors=[np.random.randn(size, rank)
                                            for size in tensor_sizes]
        elif random_method.lower() == 'uniform':
            matrices_factors = [np.random.uniform(size=(size[1], rank)) for size in matrices_sizes]
            tensor_factors=[np.random.uniform(
                size = (size, rank)) for size in tensor_sizes]
        else:
           raise ValueError(
               "`random_method` must be either 'normal' or 'uniform'")

        return cls(tensor_factors, matrices_factors, coupling_modes).normalize_components(update_weights=False)

    # @property
    # def shapes(self):
    #     return [fm.shape[0] for fm in self.factor_matrices], [fm.shape for fm in self.matrices_factors]


    def __getitem__(self, item):
        #return self.factor_matrices + self.matrices_factors
        pass
    def load_from_hdf5_group(self, cls, group):
        pass
    def store_in_hdf5_group(self, group):
        # self._prepare_hdf5_group(group)
        # group.attrs['tensor_factor_matrices'] = len(self.factor_matrices)
        # group.attrs['rank'] = self.rank
        # group.attrs['coupled_matrices_factors'] = 2*len(self.matrices_factors)
        # group['weights'] = self.tensor.weights
        pass

class CoupledTensors2(BaseDecomposedTensor):
    def __init__(self, main_tensor_factors, uncoupled_tensor_factors, coupling_modes, main_weights=None, uncoupled_weights=None):
        # tensor: the tensor-factors to be coupled, matrices: nested list of matrix-factors to couple, coupling_modes: list of modes
        if len(uncoupled_tensor_factors) != len(coupling_modes):
            raise ValueError('Coupled tensors was {0} but coupling modes was {1}'.format(len(uncoupled_tensor_factors), len(coupling_modes)))
        self.rank = main_tensor_factors[0].shape[1]
        self.coupling_modes = coupling_modes
        #self.uncoupled_tensor_factors = uncoupled_tensor_factors
        self._create_kruskals(main_tensor_factors, uncoupled_tensor_factors, main_weights=main_weights, uncoupled_weights=uncoupled_weights)
    
    @property
    def factor_matrices(self):
        return self.main_tensor.factor_matrices

    @property
    def coupled_factor_matrices(self):
        return [self.main_tensor.factor_matrices[i] for i in self.coupling_modes]

    @property
    def uncoupled_tensor_factors(self):
        uncoupled_factors = [None]*len(self.coupling_modes)
        for i, mode in enumerate(self.coupling_modes):
            if len((self.coupled_tensors+self.coupled_matrices)[i].factor_matrices) > 2:
                factors = self.coupled_tensors[i].factor_matrices.copy()
                factors.pop(mode)
                uncoupled_factors[i] = factors
            else:
                #TODO: make more elegant
                uncoupled_factors[i] = (self.coupled_tensors+self.coupled_matrices)[i].factor_matrices[1]
        return uncoupled_factors

    def _create_kruskals(self, tensor_factors, uncoupled_tensor_factors, main_weights=None, uncoupled_weights=None):
        #TODO: np.copy on the coupled matrices
        main_weights = np.ones(self.rank) if main_weights is None else main_weights
        #uncoupled_weights = [np.ones(self.rank) for _ in range(len(self.coupling_modes))] if uncoupled_weights is None else uncoupled_weights
        self.main_tensor = KruskalTensor(tensor_factors, weights=main_weights)
        self.coupled_tensors = []
        self.coupled_matrices = []
        for i, mats in enumerate(uncoupled_tensor_factors):
            mode = self.coupling_modes[i]
            if len(mats) == 1:
                self.coupled_matrices.append(KruskalTensor([tensor_factors[mode], mats[0]], weights=None if uncoupled_weights is None else uncoupled_weights[i]))
            else:
                factors = [tensor_factors[mode]]+mats
                factors.insert(mode, factors.pop(0))
                self.coupled_tensors.append(KruskalTensor(factors, weights=None if uncoupled_weights is None else uncoupled_weights[i]))

    def construct_tensor(self):
        return self.main_tensor.construct_tensor()

    def construct_coupled_tensors(self):
        return [tensor.construct_tensor() for tensor in self.coupled_tensors + self.coupled_matrices]

    def reset_weights(self):
        for obj in [self.main_tensor] + self.coupled_tensors + self.coupled_matrices:
            obj.reset_weights()

    def normalize_components(self, update_weights=True, eps=1e-15):
        """Set all factor matrices to unit length. Updates the weights if `update_weights` is True.

        Arguments:
        ----------
        update_weights : bool
            If true, then the weights of this Kruskal tensor will be set to the product of the
            component norms.
        """
        for obj in [self.main_tensor] + self.coupled_tensors + self.coupled_matrices:
            obj.normalize_components(update_weights=update_weights)
        return self

    @classmethod
    def random_init(cls, main_tensor_shape, rank, coupled_tensors_shapes, coupling_modes, random_method='normal'):
        """Construct a random Kruskal tensor coupled with n matrices, all with unit vectors as components and unit weights.
        """
    #    # check that modes are correct TODO: make new exception.
    #     for i, size in enumerate(coupled_tensors_shapes):
    #         if size[0] != main_tensor_shape[coupling_modes[i]]:
    #             raise ValueError('The coupling is not right.')

        if random_method.lower() == 'normal':
            coupled_tensor_factors = []
            for mode, shape in zip(coupling_modes, coupled_tensors_shapes):
                if len(shape) > 2:
                    shape = [dim for j, dim in enumerate(shape) if mode!=j]
                    factors = [np.random.randn(size, rank) for size in shape]
                else:
                    factors = [np.random.randn(shape[1], rank)]
                coupled_tensor_factors.append(factors)
            tensor_factors=[np.random.randn(size, rank)
                                            for size in main_tensor_shape]
        elif random_method.lower() == 'uniform':
            coupled_tensor_factors = []
            for shape in coupled_tensors_shapes:
                if type(shape) != list:
                    shape = [shape]
                factors = [np.random.uniform(size=(size[1], rank)) for size in shape]
                coupled_tensor_factors.append(factors)
            tensor_factors = [np.random.uniform(size=(size, rank))
                                            for size in main_tensor_shape]
        else:
           raise ValueError(
               "`random_method` must be either 'normal' or 'uniform'")
        return cls(tensor_factors, coupled_tensor_factors, coupling_modes).normalize_components(update_weights=False)

    # @property
    # def shapes(self):
    #     return [fm.shape[0] for fm in self.factor_matrices], [fm.shape for fm in self.matrices_factors]


    def __getitem__(self, item):
        #return self.factor_matrices + self.matrices_factors
        pass
    def load_from_hdf5_group(self, cls, group):
        pass
    def store_in_hdf5_group(self, group):
        # self._prepare_hdf5_group(group)
        # group.attrs['tensor_factor_matrices'] = len(self.factor_matrices)
        # group.attrs['rank'] = self.rank
        # group.attrs['coupled_matrices_factors'] = 2*len(self.matrices_factors)
        # group['weights'] = self.tensor.weights
        pass
