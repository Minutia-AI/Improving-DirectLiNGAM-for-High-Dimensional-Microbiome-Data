import numpy as np
from sklearn.metrics import mean_squared_error
#import pandas as pd
#import graphviz
#import lingam
#from lingam.utils import print_causal_directions, print_dagc, make_dot
#import random
#import networkx as nx
#from itertools import product
#import os

def sample_from_disjoint_interval(size, rng):
    # weights in [-1.5,-0.5] ∪ [0.5,1.5]
    coin = rng.random(size) < 0.5
    left  = rng.uniform(-1.5, -0.5, size)
    right = rng.uniform( 0.5,  1.5, size)
    return np.where(coin, left, right)

def generate_connection_matrix(p, s, rng):
    """
    s: prob of an edge below the diagonal (acyclic in natural order).
    For expected in-degree ≈ 1, use s ≈ 1/(p-1).
    """
    # strictly lower triangular Bernoulli mask
    mask = rng.binomial(1, s, size=(p, p)) * np.tril(np.ones((p, p), dtype=int), k=-1)
    num_edges = int(mask.sum())
    W = np.zeros((p, p), dtype=float)
    if num_edges > 0:
        W[mask == 1] = sample_from_disjoint_interval(num_edges, rng)
    return W  # this is B

def sample_uniform_noise(n, p, variances, rng):
    """
    For each variable j, draw e_j ~ Uniform[-a_j, a_j] with Var = variances[j],
    where a_j = sqrt(3 Var).
    Vectorized implementation.
    """
    scales = np.sqrt(3.0 * variances)  # shape (p,)
    # draw U ~ Uniform[-1,1], then scale per-column
    U = rng.uniform(-1.0, 1.0, size=(n, p))
    return U * scales  # broadcasts scales over rows

def permute_matrix(B, perm):
    """Return B_perm = P^T B P for column permutation 'perm'."""
    P = np.eye(B.shape[0])[perm]           # rows permuted
    Pinv = P.T                              # permutation matrices are orthogonal
    return Pinv @ B @ P

def generate_dataset(E, B, n, permutation=None, rng=None):
    """
    Generate dataset X from external influences E and connection matrix B,
    with Gaussian shift and optional column permutation.
    
    Parameters:
    - E: (n, p) matrix of external influences
    - B: (p, p) connection strength matrix
    - permutation: list or array of length p (optional). If None, a random permutation is generated.
    
    Returns:
    - X_perm: the permuted data matrix (n, p)
    - means: Gaussian shifts applied to each variable (p,)
    - permutation: the permutation used (to apply consistently across groups)
    """
    n, p = E.shape
    I = np.eye(p)
    A_inv = np.linalg.inv(I - B).T
    X = E @ A_inv  # Step: X = E (I - B)^-1

    # Step: Add Gaussian mean shift (N(0, 4) → std = 2)
    means = rng.normal(loc=0, scale=2.0, size=p)
    X += means  # Broadcast to shift each variable

    # Step: Apply consistent variable permutation
    if permutation is None:
        permutation = rng.permutation(p)

    X_perm = X[:, permutation]  # Permute columns
    
    return X_perm, means, permutation


def simulate_data(p, s, n, seed = None, perm = None):
    rng = np.random.default_rng(seed)
    B = generate_connection_matrix(p, s, rng)
    sample_size = n
    variances = rng.uniform(1, 3, size=p)
    E = sample_uniform_noise(sample_size, p, variances, rng)
    X, _, perm = generate_dataset(E, B, sample_size, permutation=perm, rng=rng)
    
    return X, perm, B

def average_squared_error(true_B, est_B):
    # Flatten matrices and compute MSE
    return mean_squared_error(true_B.flatten(), est_B.flatten())

def invert_permutation(B, perm):
    inverse_perm = np.argsort(perm)
    B_perm_inverted = B[inverse_perm, :][:, inverse_perm]
    return B_perm_inverted
