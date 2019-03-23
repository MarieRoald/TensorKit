from abc import ABC, abstractmethod, abstractproperty
import numpy as np
# TODO: More loggers
# - gradient logger (need to implement gradient for decomposer first)

class BaseLogger(ABC):
    def __init__(self):
        self.log_values = []
        self.log_iterations = []
        self.prev_checkpoint_it = 0

    @abstractmethod
    def _log(self, decomposer):
        pass

    def log(self, decomposer):
        self._log(decomposer)
        self.log_iterations.append(decomposer.current_iteration)

    @property
    def latest_log_values(self):
        return self.log_values[self.prev_checkpoint_it:]

    @property
    def latest_log_iterations(self):
        return self.log_iterations[self.prev_checkpoint_it:]


    def _write_log_to_hd5_group(self, logger_group, logname, log):
        if logname in logger_group:
            old_length = logger_group[logname].shape[0]
            new_length = old_length + len(log)  # Noe rart skjer her tror jeg
            logger_group[logname].reshape(new_length, axis=0)
            logger_group[logname][old_length:] = log
        else:
            logger_group.create_dataset(logname, shape=(len(log),), maxshape=(None,))
            logger_group[logname] = log

        self.prev_checkpoint_it += len(log)
    
    def write_to_hdf5_group(self, h5group):
        logger_group = h5group.require_group(type(self).__name__)
        self._write_log_to_hd5_group('iterations', logger_group, self.latest_log_iterations)
        self._write_log_to_hd5_group('values', logger_group, self.latest_log_values)


class LossLogger(BaseLogger):
    def _log(self, decomposer):
        self.log_values.append(decomposer.loss)

class MSELogger(BaseLogger):
    def _log(self, decomposer):
        self.log_values.append(decomposer.MSE)

class SSELogger(BaseLogger):
    def _log(self, decomposer):
        self.log_values.append(decomposer.SSE)

class RMSELogger(BaseLogger):
    def _log(self, decomposer):
        self.log_values.append(decomposer.RMSE)

class ExplainedVarianceLogger(BaseLogger):
    def _log(self, decomposer):
        self.log_values.append(decomposer.explained_variance)

class GradientLogger(BaseLogger):
    def _log(self, decomposer):
        self.log_values.append(decomposer.gradient)


    
