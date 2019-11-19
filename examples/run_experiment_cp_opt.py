from tenkit.log import HDF5Logger, Experiment
import numpy as np
import tenkit.cp as cp
import tenkit.base
import argparse
from scipy.io import loadmat

def _cp_opt(X, rank, method="cg", max_its=1000, gtol=1e-10, init="random", logger=None):
    factors, result, initial_factors = cp.cp_opt(X, rank, method=method, max_its=max_its, gtol=gtol, init=init, logger=logger)

    return_dict = {'num_inits': result.nit }
    for i, factor in enumerate(factors):
        return_dict[f'factor_mode_{i}'] = factor

    for i, init_factor in enumerate(initial_factors):
        return_dict[f'init_factor_mode_{i}'] = init_factor
    return return_dict

def _create_cp_opt_logger(fname, ex_name, store_frequency, X, rank, init="random"):
    init_factors, _ = cp.initialize_factors(X, rank, init)
    init_loss = cp.cp_loss(init_factors, X)
    X_norm = np.linalg.norm(X.ravel())
    del init_factors

    def loss(parameters):
        return cp._cp_loss_scipy(parameters, rank, X.shape, X )
    
    def gradient(parameters):
        return np.linalg.norm(np.ravel(cp._cp_grad_scipy(parameters, rank, X.shape, X )))
    
    args = tuple()
    
    log_metrics = {
        'loss': loss,
        'gradient': gradient,
    }

    return HDF5Logger(
        fname, ex_name, store_frequency, args, **log_metrics
    )

def _cp_opt_final_eval(X, experiment_params, outputs):
    num_factors = len(X.shape)
    factors = [outputs[f'factor_mode_{i}'] for i in range(num_factors)]
    init_factors = [outputs[f'init_factor_mode_{i}'] for i in range(num_factors)]
    rank = experiment_params['rank']
    init = experiment_params['init']

    init_loss = cp.cp_loss(init_factors, X)
    X_norm = np.linalg.norm(X.ravel())
    err = X - base.ktensor(*factors)
    return {
        'final_loss': cp.cp_loss(factors, X),
        'final_gradient': np.linalg.norm(np.concatenate(cp.cp_grad(factors, X))),
        'final_cp_opt_fit': 1 - cp.cp_loss(factors, X)/init_loss,
        'final_cp_als_fit': 1 - np.linalg.norm(err.ravel())/X_norm,
    }

def run_cp_opt(fname, num_runs, store_frequency, X, rank, max_its=1000, gtol=1e-10, init="random"):
    for run in range(num_runs):
        ex_name = f"cp_als_rank{rank}_run{run}"
        logger = _create_cp_opt_logger(fname, ex_name, store_frequency, X, rank, init=init)
        experiment = Experiment(fname, ex_name)
        ex_params = {'rank': rank, 'max_its': max_its, 'gtol':gtol, 'init': init}
        experiment.run_experiment(X, _cp_opt, ex_params, _cp_opt_final_eval, logger=logger)
        logger.save_logs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str)
    parser.add_argument("num_runs", type=int)
    parser.add_argument("store_frequency", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("data_location", type=str)
    parser.add_argument("tensor_name", type=str)
    parser.add_argument("--max_its", default=1000, type=int)
    parser.add_argument("--gtol", default=1e-10, type=float)
    parser.add_argument("--init", default="random", type=str)
    args = parser.parse_args()
    X = loadmat(args.data_location)[args.tensor_name]
    X = np.ascontiguousarray(X, dtype=X.dtype)
    

    run_cp_opt(fname=args.fname, num_runs=args.num_runs, store_frequency=args.store_frequency, X=X, rank=args.rank,
               max_its=args.max_its, gtol=args.gtol, init=args.init)