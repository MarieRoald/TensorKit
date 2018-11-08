import numpy as np
import itertools
import base
import cp
import scipy
import utils
def weight_score(weight1, weight2):
    return np.abs(weight1-weight2)/max(weight1, weight2)

def _factor_match_score(true_factors, estimated_factors, weight_penalty=True):
    rank = true_factors[0].shape[1]

    # Make sure columns of factor matrices are normalized
    true_factors, true_norms = utils.normalize_factors(true_factors)
    estimated_factors, estimated_norms = utils.normalize_factors(estimated_factors)  

    if weight_penalty:
        true_weights = np.prod(np.concatenate(true_norms), axis=0)
        estimated_weights = np.prod(np.concatenate(estimated_norms), axis=0)
    else:
        true_weights = np.ones((rank,))
        estimated_weights = np.ones((rank,))

    scores = []
    for r in range(rank):
        score = (1-weight_score(true_weights[r], estimated_weights[r]))
        for true_factor, estimated_factor in zip(true_factors, estimated_factors):
            score *= np.abs(true_factor[:,r].T@estimated_factor[:,r])
        
        scores.append(score)
    return scores
    
def factor_match_score(true_factors, estimated_factors, weight_penalty=True, fms_reduction='min'):
    if fms_reduction == 'min':
        fms_reduction = np.min
    elif fms_reduction == 'mean':
        fms_reduction = np.mean
    else:
        raise ValueError('́`fms_reduction` must be either "min" or "mean".')

    rank = true_factors[0].shape[1]

    max_fms = -1
    best_permutation = None
    
    for permutation in itertools.permutations(range(rank),r=rank):
        permuted_factors = utils.permute_factors(permutation, estimated_factors)
        
        fms = fms_reduction(_factor_match_score(true_factors, permuted_factors, weight_penalty=weight_penalty))

        if fms > max_fms:
            max_fms = fms
            best_permutation = permutation
    return max_fms, best_permutation

            
def tensor_completion_score(X, X_hat, W):
    return np.linalg.norm((1-W)*(X - X_hat))/np.linalg.norm((1-W)*X)


def core_consistency(X, A, B, C):
    F = A.shape[1]
    F = A.shape[1]

    k = np.kron(A.T,np.kron(B.T,C.T))
    #k = base.kron(A.T,B.T,C.T)

    T = np.zeros((F,F,F))
    np.fill_diagonal(T, 1)

    vec_T = T.reshape((-1, 1))
    vec_X = X.reshape((-1, 1))
    vec_G = np.linalg.pinv(k).T@vec_X

    return 100*(1-sum((vec_G-vec_T)**2)/F)
