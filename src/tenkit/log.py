from math import ceil

import h5py
import numpy as np

from . import base


"""
class Logger:
    def __init__(self, args, loss, grad):
        self.loss_values = []
        self.gradients = []
        self.gradient_values = []
        self.args = args

        self.loss = loss
        self.grad = grad

    def log(self, parameters):

        loss = self.loss(parameters, *self.args)

        grad = self.grad(parameters, *self.args)
        gradients = base.unflatten_factors(grad, *self.args[:2])

        self.loss_values.append(loss)
        self.gradients.append(gradients)
        self.gradient_values.append(np.linalg.norm(grad))
"""


class Logger:
    def __init__(self, args, **targets):
        self.args = args
        self.target_functions = targets
        self.target_values = {target: [] for target in targets}
    
    def log(self, parameters):
        for target, values in self.target_values.items():
            f = self.target_functions[target]
            values.append(f(parameters, *self.args))
    

class HDF5Logger(Logger):
    def __init__(self, fname, ex_name, store_frequency, args, continue_old=False, **targets):
        """Log data to a HDF5 file.

        Parameters:
        -----------
        fname : str
            Filename
        ex_name : str
            Name of experiment, used as the group to store the experiment parameters in.
        store_frequency : int
            How often to store the results to disc
        args : tuple
            Tuple of arguments to feed the target functions
        **targets : functions
            The functions that we want to log.
        """
        super().__init__(args, **targets)
        self.fname = fname
        self.ex_name = ex_name
        self.store_frequency = store_frequency
        if not continue_old:
            self._init_h5_file()
            self._it_num = 0
        else:
            with h5py.File(self.fname, 'r') as h5:
                target = list(targets.keys())[0]
                self._it_num = len(h5[ex_name][target])
        self._prev_write_it = self._it_num
        self._first_write_it = self._prev_write_it

    def _init_h5_file(self):
        with h5py.File(self.fname, 'a') as h5:
            if self.ex_name not in h5:
                g = h5.create_group(self.ex_name)
            else:
                g = h5[self.ex_name]

            for target in self.target_functions:
                g.create_dataset(target, dtype=np.float32, shape=[0], maxshape=[None])

    def log(self, parameters):
        super().log(parameters)
        self._it_num += 1
        if self._it_num % self.store_frequency == 0:
            self.save_logs()
    
    def save_logs(self):
        with h5py.File(self.fname, 'a') as h5:
            g = h5[self.ex_name]
            for target, values in self.target_values.items(): 
                g[target].resize([self._it_num])
                g[target][self._prev_write_it:] = values[self._prev_write_it-self._first_write_it:] 
        self._prev_write_it = self._it_num

class Experiment:
    def __init__(self, fname, ex_name, attributes=None, metadata=None):
        self.fname = fname
        self.ex_name = ex_name
        self._init_h5_file(attributes, metadata)

    def _init_h5_metadata(self, h5, metadata):
        if 'metadata' not in h5:
            g = h5.create_group('metadata')
        else:
            g = h5['metadata']

        for name, value in metadata.items():
            g[name] = value
    
    def _init_h5_file(self, attributes, metadata):
        with h5py.File(self.fname, 'a') as h5:
            if self.ex_name not in h5:
                g = h5.create_group(self.ex_name)
            else:
                g = h5[self.ex_name]

            if attributes is not None:
                for key, value in attributes.items():
                    g.attrs[key] = value
            
            if metadata is not None:
                self._init_h5_metadata(h5, metadata)

    def run_experiment(self, X, experiment_function, experiment_params, final_eval, 
                       checkpoint_freq=200, logger=None, continue_old=False):
        """
        Stores key-value pairs of outputs = experiment_function(**experiment_params) and 
        final_eval_metrics(experiment_params, outputs) as elements of the hdf5 file.

        To run experiment: define the experiment_function and final_eval_metrics.

        Parameters:
        -----------
        experiment_function : function
            Returns dict
        experiment_params : dict
            kwargs for experiment_function
        final_eval : function
            Takes two dicts as input, experiment_params and experiment_function(**experiment_params)
            Returns dict
        continue_old : bool
            If this is true, the init-parameter are not written to the experiment attributes.
        """
        #if 'max_its' not in experiment_params and checkpoint_saver is not None:
        #    raise ValueError('Cannot checkpoint code if max_its is not set')
        #elif 'max_its' in experiment_params and checkpoint_saver is not None:
        #    max_its = experiment_params['max_its']
        #    num_runs = ceil(max_its/checkpoint_freq)
        #    experiment_params['max_its'] = checkpoint_freq

        with h5py.File(self.fname, 'a') as h5:
            outputs = experiment_function(X, **experiment_params, 
                                          h5_group=h5[self.ex_name], logger=logger, 
                                          load_old=continue_old)

        with h5py.File(self.fname, 'a') as h5:
            g = h5[self.ex_name]
            for param, value in experiment_params.items():
                if param == 'init':
                    continue
                g.attrs[param] = value
            for pname, param in outputs.items():
                if not continue_old:
                    g[pname] = param
                else:
                    g[pname][...] = param
            
        final_eval_metrics = final_eval(X, experiment_params, outputs)
        with h5py.File(self.fname, 'a') as h5:
            g = h5[self.ex_name]
            for pname, param in final_eval_metrics.items():
                if not continue_old:
                    g[pname] = param
                else:
                    g[pname][...] = param
