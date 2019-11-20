#TODO: Let us tidu up this a bit
import itertools

import numpy as np
import scipy

from . import base, utils


def weight_score(weight1, weight2):
    return np.abs(weight1 - weight2) / max(weight1, weight2)


def _tucker_congruence(A1, A2):      
    A1_normalised = A1/np.linalg.norm(A1, axis=0)
    A2_normalised = A2/np.linalg.norm(A2, axis=0)
    return A1_normalised.T@A2_normalised                             

def _factor_match_score(true_factors, estimated_factors, weight_penalty=True, nonnegative=True):

    if len(true_factors[0].shape) == 1:
        true_factors = [factor.reshape(-1,1) for factor in true_factors]
    if len(estimated_factors[0].shape) == 1:
        estimated_factors =  [factor.reshape(-1,1) for factor in estimated_factors] 
    
 
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
        score = 1 - weight_score(true_weights[r], estimated_weights[r])
        for true_factor, estimated_factor in zip(true_factors, estimated_factors):
            if nonnegative:
                score *= np.abs(true_factor[:, r].T @ estimated_factor[:, r])
            else:
                score *= true_factor[:, r].T @ estimated_factor[:, r]

        scores.append(score)
    return scores

def factor_match_score(
    true_factors, estimated_factors, weight_penalty=True, fms_reduction="min"
):
    if fms_reduction == "min":
        fms_reduction = np.min
    elif fms_reduction == "mean":
        fms_reduction = np.mean
    else:
        raise ValueError('`fms_reduction` must be either "min" or "mean".')

    rank = true_factors[0].shape[1]
    estimated_rank = estimated_factors[0].shape[1]

    max_fms = -1
    best_permutation = None

    for permutation in itertools.permutations(range(estimated_rank), r=rank):
        permuted_factors = utils.permute_factors(permutation, estimated_factors)

        fms = fms_reduction(
            _factor_match_score(
                true_factors, permuted_factors, weight_penalty=weight_penalty
            )
        )

        if fms > max_fms:
            max_fms = fms
            best_permutation = permutation
    return max_fms, best_permutation

def separate_mode_factor_match_score(true_factors, estimated_factors, fms_reduction='min'):
    if fms_reduction == "min":
        fms_reduction = np.min
    elif fms_reduction == "mean":
        fms_reduction = np.mean
    else:
        raise ValueError('`fms_reduction` must be either "min" or "mean".')

    rank = true_factors[0].shape[1]
    estimated_rank = estimated_factors[0].shape[1]


    max_fms = []
    best_permutation = []

    for true_factor, estimated_factor in zip(true_factors, estimated_factors):
        current_max_fms = -1
        current_best_permutation = None
        for permutation in itertools.permutations(range(estimated_rank), r=rank):
            permuted_factor = utils.permute_factors(permutation, [estimated_factor])

            fms = fms_reduction(
                _factor_match_score(
                    [true_factor], permuted_factor, weight_penalty=False
                )
            )

            if fms > current_max_fms:
                current_max_fms = fms
                current_best_permutation = permutation
        
        max_fms.append(current_max_fms)
        best_permutation.append(current_best_permutation)
    return max_fms, best_permutation



def tensor_completion_score(X, X_hat, W):
    return np.linalg.norm((1 - W) * (X - X_hat)) / np.linalg.norm((1 - W) * X)


def core_consistency(X, A, B, C, normalized=False):
    # TODO: generalise to more than three modes
    F = A.shape[1]








    # Create the superdiagonal tensor
    T = np.zeros((F,F,F))
    np.fill_diagonal(T, 1)
    
    # Separate the weights evenly along the three modes
    [A, B, C], norms = utils.normalize_factors((A, B, C))
    weights = np.prod(np.concatenate(norms), axis=0, keepdims=True)
    A *= weights**(1/3)
    B *= weights**(1/3)
    C *= weights**(1/3)

    # Generate vectorized variables to find Tucker core    
    k = base.kron(A.T, B.T, C.T)
    vec_T = T.reshape((-1, 1))
    vec_X = X.reshape((-1, 1))

    # ALS-step to find Tucker core
    Q, R = np.linalg.qr(k.T)
    _rhs = Q.T @ vec_X
    vec_G = scipy.linalg.solve_triangular(R, _rhs)

    # Old versions
    # _rhs = k @ vec_X
    # _rhs = solve_triangular(R, _rhs, trans='T')
    #   vec_G = np.linalg.lstsq(k.T, vec_X)[0]

    denom = np.linalg.norm(vec_G)**2 if normalized else F
    return np.squeeze(100*(1-sum((vec_G-vec_T)**2)/denom))



def core_consistency_parafac2(X, P_k, A, F, C):
    rank = F.shape[1]
    I = A.shape[0]
    K = C.shape[0]
    projected_X = np.empty((I, rank, K))

    for k, projection_matrix in enumerate(P_k):
        projected_X[..., k] = X[k]@projection_matrix

    return core_consistency(projected_X, A, F, C)



"""
def calculate_core_consistencies(X, upper_rank=5):
    core_consistencies = []
    for k in range(1, upper_rank + 1):
        factors, result, initial_factors = cp.cp_opt(
            X, rank=k, method="cg", init="svd", gtol=1e-10
        )
        A, B, C = factors
        c = core_consistency(X, A, B, C)
        core_consistencies.append(c)
    return core_consistencies
"""


def leverage(factor_matrix):
    leverage_scores = np.diagonal(factor_matrix@np.linalg.inv((factor_matrix.T@factor_matrix)).T@factor_matrix.T)
    return leverage_scores


def _factor_match_score_parafac2(true_factors, estimated_factors, weight_penalty=True, nonnegative=True):

    if len(true_factors[0].shape) == 1:
        true_factors = [factor.reshape(-1,1) for factor in true_factors]
    if len(estimated_factors[0].shape) == 1:
        estimated_factors =  [factor.reshape(-1,1) for factor in estimated_factors] 
    
 
    rank = true_factors[0].shape[1]

    # Make sure columns of factor matrices are normalized
    # true_factors, true_norms = utils.normalize_factors(true_factors)
    # estimated_factors, estimated_norms = utils.normalize_factors(estimated_factors)

    true_factors = [
        true_factors[0]/np.linalg.norm(true_factors[0], axis=0),
        [f/np.linalg.norm(f, axis=0) for f in true_factors[1]],
        true_factors[2]/np.linalg.norm(true_factors[2], axis=0)
    ]
    true_factors = [
        estimated_factors[0]/np.linalg.norm(estimated_factors[0], axis=0),
        [f/np.linalg.norm(f, axis=0) for f in estimated_factors[1]],
        estimated_factors[2]/np.linalg.norm(estimated_factors[2], axis=0)
    ]

    if weight_penalty:
        raise NotImplementedError
        true_weights = np.prod(np.concatenate(true_norms), axis=0)
        estimated_weights = np.prod(np.concatenate(estimated_norms), axis=0)
    else:
        true_weights = np.ones((rank,))
        estimated_weights = np.ones((rank,))

    scores = []
    for r in range(rank):
        score = 1 - weight_score(true_weights[r], estimated_weights[r])

        for true_factor, estimated_factor in zip(true_factors[0:3:2], estimated_factors[0:3:2]):
            if nonnegative:
                score *= np.abs(true_factor[:, r].T @ estimated_factor[:, r])
            else:
                score *= true_factor[:, r].T @ estimated_factor[:, r]

        true_evolving_factor = true_factors[1]
        estimated_evolving_factor = estimated_factors[1]

        evolving_score = 1

        for true_factor, estimated_factor in zip(true_evolving_factor, estimated_evolving_factor):
            if nonnegative:
                evolving_score *= np.abs(true_factor[:, r].T @ estimated_factor[:, r])
            else:
                evolving_score *= true_factor[:, r].T @ estimated_factor[:, r]

        K = len(true_evolving_factor)

        evolving_score = evolving_score**(1/K)
        score *= evolving_score

        scores.append(score)
    return scores


def factor_match_score_parafac2(
    true_factors, estimated_factors, weight_penalty=True, fms_reduction="min"
):
    if fms_reduction == "min":
        fms_reduction = np.min
    elif fms_reduction == "mean":
        fms_reduction = np.mean
    else:
        raise ValueError('́`fms_reduction` must be either "min" or "mean".')

    rank = true_factors[0].shape[1]
    estimated_rank = estimated_factors[0].shape[1]

    max_fms = -1
    best_permutation = None
    true_factors = [true_factors[0], true_factors[2]]

    for permutation in itertools.permutations(range(estimated_rank), r=rank):
        permuted_factors = [
            estimated_factors[0][:, permutation],
            estimated_factors[2][:, permutation]
        ]

        fms = fms_reduction(
            _factor_match_score(
                true_factors, permuted_factors, weight_penalty=weight_penalty
            )
        )

        if fms > max_fms:
            max_fms = fms
            best_permutation = permutation
    return max_fms, best_permutation


def percent_explained(true_tensor, estimated_tensor):
    SSE = np.linalg.norm(true_tensor-estimated_tensor)**2
    SSX = np.linalg.norm(true_tensor)**2
    return 1 - SSE/SSX
