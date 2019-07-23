from pathlib import Path
import tempfile

import h5py
import pytest
import numpy as np
from .. import cp
from .. import decompositions
from ... import metrics
from .. import cmtf2


# these tests take forever :thinking:
class TestCMTF2ALS:

    def test_nan_imputation(self):

        pass

    def test_rank3_tensor_with_one_coupled_matrix(self):
        shape = (60, 40, 100)
        rank = 3
        A = np.random.standard_normal((shape[0],rank))
        B = np.random.standard_normal((shape[1],rank))
        C = np.random.standard_normal((shape[2],rank))
        V = np.random.standard_normal((20,rank))

        X = decompositions.KruskalTensor([A, B, C]).construct_tensor()
        Y = A @ V.T
        cmtf = cmtf2.CMTF_ALS(rank=rank, max_its=3000, print_frequency=100)
        decomp = cmtf.fit_transform(X=X, coupled_matrices=[Y], coupling_modes=[0])
        estimated_X = decomp.construct_tensor()
        estimated_A, estimated_V = decomp.factor_matrices[0], decomp.uncoupled_factor_matrices[0]
        estimated_Y = estimated_A @ estimated_V.T
        assert np.allclose(X, estimated_X)
        assert np.allclose(Y, estimated_Y)
    
    def test_rank4_tensor_with_two_coupled_matrices_on_mode_0(self):
        shape = (90, 40, 86)
        rank = 3
        A = np.random.standard_normal((shape[0],rank))
        B = np.random.standard_normal((shape[1],rank))
        C = np.random.standard_normal((shape[2],rank))
        V1 = np.random.standard_normal((20,rank))
        V2 = np.random.standard_normal((50,rank))

        X = decompositions.KruskalTensor([A, B, C]).construct_tensor()
        Y1 = A @ V1.T
        Y2 = A @ V2.T
        cmtf = cmtf2.CMTF_ALS(rank=rank, max_its=3000, print_frequency=100)
        decomp = cmtf.fit_transform(X=X, coupled_matrices=[Y1, Y2], coupling_modes=[0, 0])

        estimated_X = decomp.construct_tensor()
        estimated_A, estimated_V1, estimated_V2 = decomp.factor_matrices[0], decomp.uncoupled_factor_matrices[0], decomp.uncoupled_factor_matrices[1]
        estimated_Y1 = estimated_A @ estimated_V1.T
        estimated_Y2 = estimated_A @ estimated_V2.T
        assert np.allclose(X, estimated_X)
        assert np.allclose(Y1, estimated_Y1)
        assert np.allclose(Y2, estimated_Y2)
        
    def test_rank3_tensor_with_two_coupled_matrices_on_different_modes(self):
        shape = (90, 40, 86)
        rank = 3
        A = np.random.standard_normal((shape[0],rank))
        B = np.random.standard_normal((shape[1],rank))
        C = np.random.standard_normal((shape[2],rank))
        V1 = np.random.standard_normal((20,rank))
        V2 = np.random.standard_normal((50,rank))
        X = decompositions.KruskalTensor([A, B, C]).construct_tensor()
        Y1 = A @ V1.T
        Y2 = C @ V2.T

        cmtf = cmtf2.CMTF_ALS(rank=rank, max_its=3000, print_frequency=100)
        decomp = cmtf.fit_transform(X=X, coupled_matrices=[Y1, Y2], coupling_modes=[0, 2])

        estimated_X = decomp.construct_tensor()
        estimated_A, estimated_C = decomp.factor_matrices[0],decomp.factor_matrices[2] 
        estimated_V1, estimated_V2 = decomp.uncoupled_factor_matrices[0], decomp.uncoupled_factor_matrices[1]
        estimated_Y1 = estimated_A @ estimated_V1.T
        estimated_Y2 = estimated_C @ estimated_V2.T
        assert np.allclose(X, estimated_X)
        assert np.allclose(Y1, estimated_Y1)
        assert np.allclose(Y2, estimated_Y2)

    def test_rank3_tensor_with_three_coupled_matrices(self):
        shape = (77, 40, 86, 50)
        rank = 3
        A = np.random.standard_normal((shape[0],rank))
        B = np.random.standard_normal((shape[1],rank))
        C = np.random.standard_normal((shape[2],rank))
        D = np.random.standard_normal((shape[3],rank))
        V1 = np.random.standard_normal((20,rank))
        V2 = np.random.standard_normal((50,rank))
        V3 = np.random.standard_normal((15,rank))
        X = decompositions.KruskalTensor([A, B, C, D]).construct_tensor()
        Y1 = A @ V1.T
        Y2 = A @ V2.T
        Y3 = C @ V3.T

        cmtf = cmtf2.CMTF_ALS(rank=rank, max_its=100, print_frequency=10)
        decomp = cmtf.fit_transform(X=X, coupled_matrices=[Y1, Y2, Y3], coupling_modes=[0, 0, 2])

        estimated_X = decomp.construct_tensor()
        estimated_A, estimated_C = decomp.factor_matrices[0],decomp.factor_matrices[2] 
        estimated_V1, estimated_V2 = decomp.uncoupled_factor_matrices[0], decomp.uncoupled_factor_matrices[1]
        estimated_V3 = decomp.uncoupled_factor_matrices[2]
        estimated_Y1 = estimated_A @ estimated_V1.T
        estimated_Y2 = estimated_A @ estimated_V2.T
        estimated_Y3 = estimated_C @ estimated_V3.T
        assert np.allclose(X, estimated_X)
        assert np.allclose(Y1, estimated_Y1)
        assert np.allclose(Y2, estimated_Y2)
        assert np.allclose(Y3, estimated_Y3)