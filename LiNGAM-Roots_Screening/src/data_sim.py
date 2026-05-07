import numpy as np

def sample_from_disjoint_interval(size, rng):
    # weights in [-1.5,-0.5] ∪ [0.5,1.5]
    coin = rng.random(size) < 0.5
    left  = rng.uniform(-1.5, -0.5, size)
    right = rng.uniform( 0.5,  1.5, size)
    return np.where(coin, left, right)


def generate_triangular_mask_roots_exact(
    p: int,
    s: float,
    r: int,
    *,
    rng: np.random.Generator,
):
    """
    Generate a strictly lower-triangular adjacency mask with EXACTLY r roots.
    Convention: mask[i, j] = 1 means edge j -> i (parent in column, child in row).

    Parameters
    ----------
    p : int
        Number of nodes.
    s : float
        Bernoulli probability for edges below the diagonal.
    r : int
        Desired number of roots (rows with no incoming edges).
    rng : np.random.Generator
        Random generator (reproducible if seeded by caller).

    Returns
    -------
    mask : (p, p) np.ndarray of {0,1}
        Strictly lower-triangular adjacency mask.
    roots : np.ndarray
        Indices designated as (enforced) roots (size r).

    Notes
    -----
    - Roots are enforced as the first r rows (0..r-1) by zeroing their rows.
    - Acyclicity is preserved since we only place edges with row > col.
    - Feasibility requires 0 <= r <= p.
    """
    if r < 0 or r > p:
        raise ValueError("Infeasible r relative to p (need 0 <= r <= p).")

    # 1) random strictly lower-triangular mask
    mask = rng.binomial(1, s, size=(p, p)).astype(int)
    mask = np.tril(mask, k=-1)  # zero diag and upper triangle

    # 2) enforce roots (zero rows)
    roots = np.arange(r, dtype=int)  # 0..r-1
    if r > 0:
        mask[roots, :] = 0

    # Helper: current roots (zero-row nodes)
    def current_roots(m):
        return np.where(m.sum(axis=1) == 0)[0]

    # 3) fix unintended extra roots beyond the enforced set
    #    (make sure every non-enforced row j has at least one parent i < j)
    extra = [j for j in current_roots(mask) if j not in roots]
    for j in extra:
        # candidates are any columns i < j (lower-triangular ensures acyclicity)
        candidates = np.arange(j, dtype=int)
        if candidates.size == 0:
            # j == 0 can only be a root; but if j==0 is beyond enforced roots, it's infeasible
            raise RuntimeError(
                f"Infeasible: row {j} has no possible parent (j==0). "
                f"Try increasing r to include 0 among enforced roots."
            )
        i = rng.choice(candidates)
        mask[j, i] = 1  # add one parent, j is no longer a root

    # 4) sanity checks
    cur_roots = current_roots(mask)
    if len(cur_roots) != r or not np.all(np.isin(roots, cur_roots)):
        raise AssertionError("Failed to enforce exactly r roots.")

    # strictly lower-triangular check
    if not np.all(np.triu(mask, k=0) == 0):
        raise AssertionError("Mask is not strictly lower triangular.")

    return mask, roots

def generate_connection_matrix(
    p: int,
    s: float,
    root_frac: float,
    rng: np.random.Generator,
):
    """
    Generate a weighted connection matrix W for a DAG using the triangular mask approach,
    enforcing a fixed fraction of roots and leaves.

    Parameters
    ----------
    p : int
        Number of nodes.
    s : float
        Sparsity of the graph (percentage of possible edges).
    root_frac : float
        Fraction of nodes to force as root nodes (no incoming edges).
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    W : np.ndarray (p x p)
        Weighted adjacency matrix of the generated DAG.
    """


    # Compute number of roots and leaves
    r = int(p * root_frac)
    

    # --- step 1: generate triangular mask with exact roots + leaves ---
    mask, roots = generate_triangular_mask_roots_exact(
        p=p, s=s, r=r, rng=rng
    )

    # --- step 2: convert mask to weighted adjacency matrix ---
    num_edges = int(mask.sum())
    W = np.zeros((p, p), dtype=float)

    if num_edges > 0:
        W[mask == 1] = sample_from_disjoint_interval(
            num_edges,
            rng
        )

    return W, roots


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


def simulate_data(
    p: int,
    s: float,
    n: int,
    *,
    root_frac: float,
    seed: int | None = None,
    perm: np.ndarray | None = None,
):
    """
    Simulate data X ~ LiNGAM with a generated DAG.

    Parameters
    ----------
    p : int
        Number of variables.
    s : float
        probability of an edge to exist  [0, 1].
    n : int
        Sample size.
    root_frac : float
        Fraction of nodes forced to be roots (no incoming edges).
    seed : int, optional
        RNG seed.
    perm : array-like, optional
        Optional permutation to apply in generate_dataset. Use it only if you want to fix the permutation

    Returns
    -------
    X : np.ndarray, shape (n, p)
        Simulated data matrix.
    perm : np.ndarray
        The permutation actually used by generate_dataset.
    W : np.ndarray, shape (p, p)
        Weighted adjacency (connection) matrix (B).
    roots : np.ndarray
        Indices of enforced root nodes.
    leaves : np.ndarray
        Indices of enforced leaf nodes.
    """
    rng = np.random.default_rng(seed)

    # 1) Generate weighted DAG (connection matrix) + node sets
    W, roots = generate_connection_matrix(
        p=p,
        s=s,
        root_frac=root_frac,
        rng=rng,
    )

    # 2) Sample non-Gaussian/noise terms (or your preferred noise)
    variances = rng.uniform(1.0, 3.0, size=p)
    E = sample_uniform_noise(n, p, variances, rng)  # your existing helper

    # 3) Generate dataset from model X = (I - W^T)^{-1} E (handled inside)
    X, _, perm = generate_dataset(E, W, n, permutation=perm, rng=rng)

    return X, perm, W, roots


def invert_permutation(B, perm):
    inverse_perm = np.argsort(perm)
    B_perm_inverted = B[inverse_perm, :][:, inverse_perm]
    return B_perm_inverted