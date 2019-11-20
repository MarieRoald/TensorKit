from abc import ABC, abstractmethod, abstractproperty

import numpy as np

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


class LossLogger(BaseLogger):
    def _log(self, decomposer):
        self.log_metrics.append(decomposer.loss)

class MSELogger(BaseLogger):
    def _log(self, decomposer):
        self.log_metrics.append(decomposer.MSE)

class SSELogger(BaseLogger):
    def _log(self, decomposer):
        self.log_metrics.append(decomposer.SSE)

class RMSELogger(BaseLogger):
    def _log(self, decomposer):
        self.log_metrics.append(decomposer.RMSE)

class ExplainedVarianceLogger(BaseLogger):
    def _log(self, decomposer):
        self.log_metrics.append(decomposer.explained_variance)

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
