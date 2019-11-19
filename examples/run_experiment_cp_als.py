from tenkit.log import HDF5Logger, Experiment
import numpy as np
import tenkit.cp as cp
import tenkit.base
import argparse
from scipy.io import loadmat

def _cp_als(X, rank, max_its=1000, convergence_th=1e-10, init="random", logger=None):
    factors, weights = cp.cp_als(X, rank, max_its, convergence_th, init, logger=logger)
    return_dict = {'weights': weights}
    for i, factor in enumerate(factors):
        return_dict[f'factor_mode_{i}'] = factor
    return return_dict

def _create_cp_logger(fname, ex_name, store_frequency, X, rank, init="random"):
    init_factors, _ = cp.initialize_factors(X, rank, init)
    init_loss = cp.cp_loss(init_factors, X)
    X_norm = np.linalg.norm(X.ravel())
    del init_factors

    def loss(factors):
        return cp.cp_loss(factors, X)
    
    def gradient(factors):
        return np.linalg.norm(np.concatenate(cp.cp_grad(factors, X)))
    
    def cp_opt_fit(factors):
        return 1 - cp.cp_loss(factors, X)/init_loss
    
    def cp_als_fit(factors):
        err = X - base.ktensor(*factors)
        return 1 - np.linalg.norm(err.ravel())/X_norm
    
    args = tuple()
    
    log_metrics = {
        'loss': loss,
        'gradient': gradient,
        'cp_opt_fit': cp_opt_fit,
        'cp_als_fit': cp_als_fit,
    }

    return HDF5Logger(
        fname, ex_name, store_frequency, args, **log_metrics
    )

def _cp_als_final_eval(X, experiment_params, outputs):
    num_factors = len(outputs) - 1
    weights = outputs['weights']
    factors = [outputs[f'factor_mode_{i}'] for i in range(num_factors)]
    factors = [f*weights[np.newaxis]**(1/len(factors)) for f in factors]
    rank = experiment_params['rank']
    init = experiment_params['init']

    init_factors, _ = cp.initialize_factors(X, rank, init)
    init_loss = cp.cp_loss(init_factors, X)
    X_norm = np.linalg.norm(X.ravel())
    err = X - base.ktensor(*factors)
    return {
        'final_loss': cp.cp_loss(factors, X),
        'final_gradient': np.linalg.norm(np.concatenate(cp.cp_grad(factors, X))),
        'final_cp_opt_fit': 1 - cp.cp_loss(factors, X)/init_loss,
        'final_cp_als_fit': 1 - np.linalg.norm(err.ravel())/X_norm,
    }

def run_cp_als(fname, num_runs, store_frequency, X, rank, max_its=1000, convergence_th=1e-10, init="random"):
    for run in range(num_runs):
        ex_name = f"cp_als_rank{rank}_run{run}"
        logger = _create_cp_logger(fname, ex_name, store_frequency, X, rank, init=init)
        experiment = Experiment(fname, ex_name)
        ex_params = {'rank': rank, 'max_its': max_its, 'convergence_th':convergence_th, 'init': init}
        experiment.run_experiment(X, _cp_als, ex_params, _cp_als_final_eval, logger=logger)
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
    parser.add_argument("--convergence_th", default=1e-10, type=float)
    parser.add_argument("--init", default="random", type=str)
    args = parser.parse_args()
    X = loadmat(args.data_location)[args.tensor_name]

    run_cp_als(fname=args.fname, num_runs=args.num_runs, store_frequency=args.store_frequency, X=X, rank=args.rank,
               max_its=args.max_its, convergence_th=args.convergence_th, init=args.init)