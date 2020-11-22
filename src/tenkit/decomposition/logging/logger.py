from abc import ABC, abstractmethod, abstractproperty
import time

import h5py
import numpy as np

from ..decompositions import EvolvingTensor
from ...metrics import factor_match_score

# TODO: More loggers
# - gradient logger (need to implement gradient for decomposer first)

class BaseLogger(ABC):
    def __init__(self):
        self.log_metrics = []
        self.log_iterations = []
        self.prev_checkpoint_it = 0

    @abstractmethod
    def _log(self, decomposer):
        pass

    def log(self, decomposer):
        """Logs metric and iterations by appending them to lists."""
        self._log(decomposer)
        self.log_iterations.append(decomposer.current_iteration)

    @property
    def latest_log_metrics(self):
        return self.log_metrics[self.prev_checkpoint_it:]

    @property
    def latest_log_iterations(self):
        return self.log_iterations[self.prev_checkpoint_it:]


    def _write_sequence_to_hd5_group(self, logname, logger_group, log):
        """Writes list of log values to HDF5 group.
        
        Arguments
        ---------
        logname: string
            Name of log. Used as name for a HDF5 dataset.
        logger_group: h5.Group
            Group to write the log to.
        log: list(int)
            List containing the log values.
        """
        log = np.array(log)
        if logname in logger_group:
            old_length = logger_group[logname].shape[0]
            new_length = old_length + len(log) 
            logger_group[logname].resize(new_length, axis=0)
            logger_group[logname][old_length:] = log
        else:
            logger_group.create_dataset(logname, shape=(len(log),), maxshape=(None,), dtype=log.dtype)
            logger_group[logname][...] = log


    def write_to_hdf5_group(self, h5group):
        """Writes log metrics and log iterations to HDF5 group."""
        logger_group = h5group.require_group(type(self).__name__)
        self._write_sequence_to_hd5_group('iterations', logger_group, self.latest_log_iterations)
        self._write_sequence_to_hd5_group('values', logger_group, self.latest_log_metrics)
        self.prev_checkpoint_it += len(self.latest_log_iterations)


class MatrixLogger(BaseLogger):
    def _write_sequence_to_hd5_group(self, logname, logger_group, log):
        """Writes list of log values to HDF5 group.
        
        Arguments
        ---------
        logname: string
            Name of log. Used as name for a HDF5 dataset.
        logger_group: h5.Group
            Group to write the log to.
        log: list(int)
            List containing the log values.
        """
        log = np.array(log)
        if len(log) == 0:
            return
        if logname in logger_group:
            old_length = logger_group[logname].shape[0]
            new_length = old_length + len(log)
            logger_dataset = logger_group[logname]
            
            oldshape = logger_dataset.shape
            shape = (new_length,)
            if hasattr(log[0], '__len__'):
                shape = (new_length, len(log[0]))
            logger_dataset.resize(shape)
            logger_dataset[old_length:] = log
        else:
            shape = (len(log),)
            maxshape = (None,)
            if hasattr(log[0], '__len__'):
                shape = (len(log), len(log[0]))
                maxshape = (None, None)
            logger_group.create_dataset(logname, shape=shape, maxshape=maxshape, dtype=log.dtype)
            logger_group[logname][...] = log


class LossLogger(BaseLogger):
    def _log(self, decomposer):
        self.log_metrics.append(decomposer.loss)

class MSELogger(BaseLogger):
    def _log(self, decomposer):
        self.log_metrics.append(decomposer.MSE)

class SSELogger(BaseLogger):
    def _log(self, decomposer):
        self.log_metrics.append(decomposer.SSE)

class RelativeSSELogger(BaseLogger):
    def _log(self, decomposer):
        self.log_metrics.append(decomposer.SSE / decomposer.X_norm)

class RMSELogger(BaseLogger):
    def _log(self, decomposer):
        self.log_metrics.append(decomposer.RMSE)

class ExplainedVarianceLogger(BaseLogger):
    def _log(self, decomposer):
        self.log_metrics.append(decomposer.explained_variance)

class RegularisationPenaltyLogger(BaseLogger):
    def _log(self, decomposer):
        self.log_metrics.append(decomposer.regularisation_penalty)

class CouplingErrorLogger(BaseLogger):
    def __init__(self, not_flexible_ok=False):
        super().__init__()
        self.not_flexible_ok = not_flexible_ok

    def _log(self, decomposer):
        if self.not_flexible_ok and not hasattr(decomposer, 'coupling_error'):
            self.log_metrics.append(1)
            return
        self.log_metrics.append(decomposer.coupling_error)

class Parafac2ErrorLogger(BaseLogger):
    def __init__(self, not_flexible_ok=False):
        super().__init__()
        self.not_flexible_ok = not_flexible_ok

    def _log(self, decomposer):
        if self.not_flexible_ok and not hasattr(decomposer, 'parafac2_error'):
            self.log_metrics.append(1)
            return
        self.log_metrics.append(decomposer.parafac2_error)


class EvolvingTensorFMSLogger(BaseLogger):
    def __init__(self, path, internal_path=None, fms_options=None, decomposition=None):
        super().__init__()
        if fms_options is None:
            fms_options = {}
        self.fms_options = fms_options

        if decomposition is not None:
            self.true_decomposition = decomposition
            return
        
        with h5py.File(path, "r") as h5:
            if internal_path is not None and internal_path != "":
                h5 = h5[internal_path]
            self.true_decomposition = EvolvingTensor.load_from_hdf5_group(h5)
        
    
    def _log(self, decomposer):
        decomposition = EvolvingTensor.from_kruskaltensor(
            decomposer.decomposition, allow_same_class=True
        )
        fms = self.true_decomposition.factor_match_score(
            decomposition, **self.fms_options, weight_penalty=False
        )[0]
        self.log_metrics.append(fms)


class EvolvingTensorFMSALogger(BaseLogger):
    def __init__(self, path, internal_path=None, fms_options=None, decomposition=None):
        super().__init__()
        if fms_options is None:
            fms_options = {}
        self.fms_options = fms_options

        if decomposition is not None:
            self.true_A = decomposition.A
            return
        
        with h5py.File(path, "r") as h5:
            if internal_path is not None and internal_path != "":
                h5 = h5[internal_path]
            true_decomposition = EvolvingTensor.load_from_hdf5_group(h5)
        
        self.true_A = true_decomposition.A
    
    def _log(self, decomposer):
        decomposition = EvolvingTensor.from_kruskaltensor(
            decomposer.decomposition, allow_same_class=True
        )
        fms = factor_match_score(
            [self.true_A], [decomposition.A], weight_penalty=False, **self.fms_options
        )[0]
        self.log_metrics.append(fms)


class EvolvingTensorFMSBLogger(BaseLogger):
    def __init__(self, path, internal_path=None, fms_options=None, decomposition=None):
        super().__init__()
        if fms_options is None:
            fms_options = {}
        self.fms_options = fms_options

        if decomposition is not None:
            self.true_B = np.array(decomposition.B)
            return
        
        with h5py.File(path, "r") as h5:
            if internal_path is not None and internal_path != "":
                h5 = h5[internal_path]
            true_decomposition = EvolvingTensor.load_from_hdf5_group(h5)
        
        self.true_B = np.array(true_decomposition.B)
    
    def _log(self, decomposer):
        decomposition = EvolvingTensor.from_kruskaltensor(
            decomposer.decomposition, allow_same_class=True
        )
        B = np.array(decomposition.B)
        rank = B.shape[-1]

        fms = factor_match_score(
            [self.true_B.reshape(-1, rank)], [B.reshape(-1, rank)], weight_penalty=False, **self.fms_options
        )[0]
        self.log_metrics.append(fms)


class EvolvingTensorFMSCLogger(BaseLogger):
    def __init__(self, path, internal_path=None, fms_options=None, decomposition=None):
        super().__init__()
        if fms_options is None:
            fms_options = {}
        self.fms_options = fms_options

        if decomposition is not None:
            self.true_C = decomposition.C
            return
        
        with h5py.File(path, "r") as h5:
            if internal_path is not None and internal_path != "":
                h5 = h5[internal_path]
            true_decomposition = EvolvingTensor.load_from_hdf5_group(h5)
        
        self.true_C = true_decomposition.C
    
    def _log(self, decomposer):
        decomposition = EvolvingTensor.from_kruskaltensor(
            decomposer.decomposition, allow_same_class=True
        )
        fms = factor_match_score(
            [self.true_C], [decomposition.C], weight_penalty=False, **self.fms_options
        )[0]
        self.log_metrics.append(fms)


class EvolvingTensorFMSBCLogger(BaseLogger):
    def __init__(self, path, internal_path=None, fms_options=None, decomposition=None):
        super().__init__()
        if fms_options is None:
            fms_options = {}
        self.fms_options = fms_options

        if decomposition is not None:
            true_B = np.array(decomposition.B)
            true_C = decomposition.C
            self.true_BC = true_B*true_C[:, np.newaxis]
            return
        
        with h5py.File(path, "r") as h5:
            if internal_path is not None and internal_path != "":
                h5 = h5[internal_path]
            true_decomposition = EvolvingTensor.load_from_hdf5_group(h5)
        
        true_B = np.array(true_decomposition.B)
        true_C = true_decomposition.C
        self.true_BC = true_B*true_C[:, np.newaxis]
        self.fms_options = fms_options
    
    def _log(self, decomposer):
        decomposition = EvolvingTensor.from_kruskaltensor(
            decomposer.decomposition, allow_same_class=True
        )
        B = np.array(decomposition.B)
        BC = B*decomposition.C[:, np.newaxis]
        rank = BC.shape[-1]

        fms = factor_match_score(
            [self.true_BC.reshape(-1, rank)], [BC.reshape(-1, rank)], weight_penalty=False, **self.fms_options
        )[0]
        self.log_metrics.append(fms)


class TrueEvolvingTensorFitLogger(BaseLogger):
    def __init__(self, path, internal_path=None):
        super().__init__()
        with h5py.File(path, "r") as h5:
            if internal_path is not None and internal_path != "":
                h5 = h5[internal_path]
            true_decomposition = EvolvingTensor.load_from_hdf5_group(h5)
        self.true_tensor = true_decomposition.construct_tensor()
    
    def _log(self, decomposer):
        X = decomposer.decomposition.construct_tensor()
        SSE = np.linalg.norm(X - self.true_tensor)**2
        SS_true_target = np.linalg.norm(self.true_tensor)**2
        fit = 1 - SSE/SS_true_target

        self.log_metrics.append(fit)


class Parafac2ADMMDualNormLogger(BaseLogger):
    def __init__(self, not_admm_ok=False):
        super().__init__()
        self.not_admm_ok = not_admm_ok
    
    def _log(self, decomposer):
        if self.not_admm_ok and not hasattr(decomposer, 'sub_problems'):
            self.log_metrics.append(0)
            return

        admm_sub_problem = decomposer.sub_problems[1]
        if self.not_admm_ok and not hasattr(admm_sub_problem, 'dual_variables'):
            self.log_metrics.append(0)
            return
        
        dual_norm = np.linalg.norm(admm_sub_problem.dual_variables)
        self.log_metrics.append(dual_norm)


class Parafac2ADMMCouplingErrorLogger(BaseLogger):
    def __init__(self, not_admm_ok=False):
        super().__init__()
        self.not_admm_ok = not_admm_ok
    
    def _log(self, decomposer):
        if self.not_admm_ok and not hasattr(decomposer, 'sub_problems'):
            self.log_metrics.append(0)
            return

        admm_sub_problem = decomposer.sub_problems[1]
        if self.not_admm_ok and not hasattr(admm_sub_problem, 'aux_fms'):
            self.log_metrics.append(0)
            return
        
        aux_Bs = admm_sub_problem.aux_fms
        Bs = decomposer.decomposition.B

        coupling_error = np.linalg.norm([
            np.linalg.norm(aux_B - B) for aux_B, B in zip(aux_Bs, Bs)
        ])
        self.log_metrics.append(coupling_error)


class Timer(BaseLogger):
    def __init__(self):
        super().__init__()
        self.initial_time = None

    def _log(self, decomposer):
        if self.initial_time is None:
            self.initial_time = time.process_time()
            current_time = self.initial_time
        else:
            current_time = time.process_time()
        self.log_metrics.append(current_time - self.initial_time)
        

class CouplingErrorLogger(BaseLogger):
    def _log(self, decomposer):
        try:
            self.log_metrics.append(decomposer.coupling_error)
        except AttributeError:
            self.log_metrics.append(-1)


class Parafac2RhoALogger(BaseLogger):
    def _log(self, decomposer):
        try:
            self.log_metrics.append(decomposer.sub_problems[0].rho)
        except:
            self.log_metrics.append(-1)


class Parafac2RhoCLogger(MatrixLogger):
    def _log(self, decomposer):
        try:
            self.log_metrics.append(decomposer.sub_problems[2].rho)
        except:
            self.log_metrics.append([-1])


class Parafac2RhoBLogger(MatrixLogger):
    def _log(self, decomposer):
        try:
            self.log_metrics.append(decomposer.sub_problems[1]._cache['rho'])
        except:
            self.log_metrics.append([-1])


class NumSubIterationsLogger(BaseLogger):
    def __init__(self, mode):
        self.mode = mode
        super().__init__()

    def _log(self, decomposer):
        try:
            num_sub_iterations = decomposer.sub_problems[self.mode].num_its
        except AttributeError:
            num_sub_iterations = -1
        self.log_metrics.append(num_sub_iterations)

class SingleCouplingErrorLogger(BaseLogger):
    def __init__(self, error_num):
        super().__init__()
        self.error_num = error_num
        self.name = f"SingleCouplingError_{self.error_num}"

    def _log(self, decomposer):
        try:
            self.log_metrics.append(decomposer.coupling_errors[self.error_num])
        except AttributeError:
            self.log_metrics.append(-1)
