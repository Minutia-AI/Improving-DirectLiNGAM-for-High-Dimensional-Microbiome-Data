import numpy as np
import networkx as nx

def sample_uniform_noise(n, p, variances, rng):
    """
    Generate uniform noise for each variable with a specified variance.
    
    For each variable j, draws noise e_j ~ Uniform[-a_j, a_j] such that
    Var(e_j) = variances[j], where a_j = sqrt(3 * variances[j]).
    Implementation is fully vectorized.

    Args:
        n (int): Number of samples.
        p (int): Number of variables.
        variances (array-like): Array of shape (p,) specifying the target variance
            for each variable.
        rng (np.random.Generator): Random number generator used for sampling.

    Returns:
        np.ndarray: Noise matrix of shape (n, p) with per-variable uniform noise.
    """
    scales = np.sqrt(3.0 * variances)  # shape (p,)
    # draw U ~ Uniform[-1,1], then scale per-column
    U = rng.uniform(-1.0, 1.0, size=(n, p))
    return U * scales  # broadcasts scales over rows

def sample_from_disjoint_interval(size, rng):
    """
    Sample values from the union of two disjoint intervals: [-1.5, -0.5] ∪ [0.5, 1.5].

    Args:
        size (int): Number of samples to generate.
        rng (np.random.Generator): Random number generator used to draw samples.

    Returns:
        np.ndarray: Array of shape (size,) containing sampled values.
    """
    # weights in [-1.5,-0.5] ∪ [0.5,1.5]
    coin = rng.random(size) < 0.5
    left  = rng.uniform(-1.5, -0.5, size)
    right = rng.uniform( 0.5,  1.5, size)
    return np.where(coin, left, right)



def generate_ba_dag_many_roots(
    p, m=1, seed=None
):
    """
    Generate a directed Barabasi-Albert Network of p nodes

    Args:
        p (int): number of nodes (variables).
        m (int): number of edges to attach from a new node to existing nodes. Defaults to 1.
        seed (int): seed. Defaults to None.

    Returns:
        B : np.ndarray, shape (p, p)
        Weighted adjacency (connection) matrix (B).
        order: list containing a causal order.
    """
    rng = np.random.default_rng(seed)

    # Scale-free undirected graph
    G = nx.barabasi_albert_graph(p, m, seed=seed)

    # ORIGINAL BA construction order is 0..p-1
    # We REVERSE it to make late nodes become roots
    order = np.arange(p)[::-1]   # [p-1, p-2, ..., 1, 0]

    # Position mapping in the reversed order
    pos = {node: k for k, node in enumerate(order)}

    B = np.zeros((p, p))

    for u, v in G.edges():
        # Orient edges according to the reversed order
        if pos[u] < pos[v]:
            parent, child = u, v
        else:
            parent, child = v, u
        
        w = sample_from_disjoint_interval(size=1, rng=rng)
        B[child, parent] = w[0]    # parent -> child

    return B, order

def generate_dataset(E, B, rng=None):
    """
    Generate dataset X from external influences E and connection matrix B,
    applying Gaussian shifts and an optional column permutation.

    Args:
        E (array-like): Matrix of shape (n, p) containing external influences.
        B (array-like): Matrix of shape (p, p) representing connection strengths.
        rng (np.random.Generator, optional): Random number generator used to
            sample Gaussian shifts and permutations. If None, a default RNG is created.

    Returns:
        X_perm (array-like): Permuted data matrix of shape (n, p).
        means (array-like): Gaussian shifts applied to each variable, shape (p,).
        permutation (array-like): The permutation applied to the columns of X.
    """
    n, p = E.shape
    I = np.eye(p)
    A_inv = np.linalg.inv(I - B).T
    X = E @ A_inv  # Step: X = E (I - B)^-1

    # Step: Add Gaussian mean shift (N(0, 4) → std = 2)
    means = rng.normal(loc=0, scale=2.0, size=p)
    X += means  # Broadcast to shift each variable

    # Step: Apply consistent variable permutation
    permutation = rng.permutation(p)

    X_perm = X[:, permutation]  # Permute columns
    
    return X_perm, means, permutation


def simulate_data(
    p: int,
    m: int,
    n: int,
    seed: int | None = None,
):
    """
    Simulate data X ~ LiNGAM using a Barabási–Albert–based generative process.

    Args:
        p (int): Number of variables.
        m (int): Number of edges to attach from a new node to existing nodes.
        n (int): Sample size.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        X (np.ndarray): Simulated data matrix of shape (n, p).
        perm (np.ndarray): The permutation applied inside `generate_dataset`.
        B (np.ndarray): Weighted adjacency (connection) matrix of shape (p, p).
    """
    rng = np.random.default_rng(seed)

    # 1) Generate weighted DAG (connection matrix) + node sets
    B, _ = generate_ba_dag_many_roots(
        p=p,
        m=m,
        seed=seed,
    )

    # 2) Sample non-Gaussian/noise terms (or your preferred noise)
    variances = rng.uniform(1.0, 3.0, size=p)
    E = sample_uniform_noise(n, p, variances, rng)  # your existing helper

    # 3) Generate dataset from model X = (I - W^T)^{-1} E (handled inside)
    X, _, perm = generate_dataset(E, B, rng=rng)

    return X, perm, B
