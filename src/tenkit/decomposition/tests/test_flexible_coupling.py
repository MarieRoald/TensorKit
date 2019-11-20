import numpy as np

from tenkit.decomposition import decompositions
from tenkit.decomposition import flexible_coupling as fc


class TestFlexibleParafac2ALS:
   def test_coupled_matrices_als(self):
       decomp = decompositions.EvolvingTensor.random_init([40, 50, 60], 5)
       x = decomp.construct_slices()
       cm_als = fc.CoupledMatrices_ALS(5, max_its=10)
       cm_als.fit(x)
       assert cm_als.loss < 1e-15

   def test_flexible_parfac2_als(self):
       decomp = decompositions.Parafac2Tensor.random_init([40, 50, 20], 5)
       x = decomp.construct_slices()
       fp_als = fc.FlexibleParafac2_ALS(5, max_its=100, signal_to_noise=10, init='cp')
       fp_als.fit(x)   
       assert fp_als.explained_variance > 1-1e-3

   def test_flexible_parfac2_als_nn(self):
       decomp = decompositions.Parafac2Tensor.random_init([40, 50, 20], 5)
       x = decomp.construct_slices()
       x = np.maximum(x, 0)
       fp_als = fc.FlexibleParafac2_ALS(5, max_its=100, signal_to_noise=10, init='cp', non_negativity_constraints=[True, True, True])
       fp_als.fit(x)   
       assert fp_als.explained_variance > 1-1e-3
