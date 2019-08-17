from tenkit.log import HDF5Logger, Experiment
import numpy as np
import tenkit.cp as cp
import tenkit.base
import tenkit.parafac2
import argparse
from scipy.io import loadmat

def _parafac2(X, rank, max_its=1000, convergence_th=1e-10, init="random", logger=None):
    P_k, F, A, D_k = parafac2.parafac2_als(X, rank, max_its, convergence_th, init, logger=logger)
    C = np.diagonal(np.dstack(D_k))
    return_dict = {'F': F, 'A': A, 'C':C }
    for i, P_i in enumerate(P_k):
        return_dict[f'P_{i}'] = P_i
    return return_dict

def _create_parafac2_logger(fname, ex_name, store_frequency, X, rank, init="random"):

    init_P_k, init_A, init_F, init_D_k = parafac2._init_parafac2(X, rank, init_scheme=init)
    init_pred = parafac2.compose_from_parafac2_factors(init_P_k, init_F, init_A, init_D_k)
    init_loss = parafac2._parafac2_loss(X, init_pred)

    X_norm = np.linalg.norm(X.ravel())
    del init_P_k, init_A, init_F, init_D_k 

    def loss(factors):

        P_k, F, A, D_k, pred = factors
        
        return parafac2._parafac2_loss(X, pred)
    
    def cp_opt_fit(factors):

        P_k, F, A, D_k, pred = factors
        return 1 - parafac2._parafac2_loss(X, pred)/init_loss
    
    def cp_als_fit(factors):
        P_k, F, A, D_k, pred = factors
        return 1 - parafac2._SSE(X, pred)/X_norm
    
    args = tuple()
    
    log_metrics = {
        'loss': loss,
        'cp_opt_fit': cp_opt_fit,
        'cp_als_fit': cp_als_fit,
    }

    return HDF5Logger(
        fname, ex_name, store_frequency, args, **log_metrics
    )

def _parafac2_final_eval(X, experiment_params, outputs):
    C = outputs['C']
    A = outputs['A']
    F = outputs['F']

    K = C.shape[0]
    
    P_k = [outputs[f'P_{i}'] for i in range(K)]

    rank = experiment_params['rank']
    init = experiment_params['init']

    init_P_k, init_A, init_F, init_D_k = parafac2._init_parafac2(X, rank, init_scheme=init)
    init_pred = parafac2.compose_from_parafac2_factors(init_P_k, init_F, init_A, init_D_k)
    init_loss = parafac2._parafac2_loss(X, init_pred)

    X_norm = np.linalg.norm(X.ravel())


    D_k = np.zeros((rank, rank, K))
    for k in range(K):
        D_k[..., k] = np.diag(C[k])

    pred = parafac2.compose_from_parafac2_factors(P_k, F, A, D_k)  


    return {
        'final_loss':parafac2._parafac2_loss(X, pred),
        'final_cp_opt_fit': 1 - parafac2._parafac2_loss(X, pred)/init_loss,
        'final_cp_als_fit': 1 - parafac2._parafac2_loss(X, pred)/X_norm
    }

def run_parafac2(fname, num_runs, store_frequency, X, rank, max_its=1000, convergence_th=1e-10, init="random"):
    for run in range(num_runs):
        ex_name = f"parafac2_rank{rank}_run{run}"
        logger = _create_parafac2_logger(fname, ex_name, store_frequency, X, rank, init=init)
        experiment = Experiment(fname, ex_name)
        ex_params = {'rank': rank, 'max_its': max_its, 'convergence_th':convergence_th, 'init': init}
        experiment.run_experiment(X, _parafac2, ex_params, _parafac2_final_eval, logger=logger)
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
    X = np.ascontiguousarray(X, dtype=X.dtype)
    run_parafac2(fname=args.fname, num_runs=args.num_runs, store_frequency=args.store_frequency, X=X, rank=args.rank,
               max_its=args.max_its, convergence_th=args.convergence_th, init=args.init)