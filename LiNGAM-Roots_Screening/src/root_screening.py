import numpy as np
import pandas as pd
from sklearn.utils import check_array 

def compute_residual(xi, xj):
    """The residual when xi is regressed on xj."""
    return xi - (np.cov(xi, xj, bias=True)[0, 1] / np.var(xj)) * xj

def search_candidate(U):
    """
    Simplified version of _search_candidate without background knowledge.
    All variables in U are considered as candidate exogenous features (Uc).
    Vj is empty as there is no background knowledge to restrict connections.
    """
    Uc = list(U)  # all variables are candidates
    Vj = []       # no knowledge to filter sink variables
    return Uc, Vj

def entropy(u):
    """Calculate entropy using the maximum entropy approximations."""
    k1 = 79.047
    k2 = 7.4129
    gamma = 0.37457
    return (1 + np.log(2 * np.pi)) / 2 - k1 * (
        np.mean(np.log(np.cosh(u))) - gamma) ** 2 - k2 * (np.mean(u * np.exp((-(u ** 2)) / 2))) ** 2

def diff_mutual_info(xi_std, xj_std, ri_j, rj_i):
    """Calculate the difference of the mutual informations."""
    return (entropy(xj_std) + entropy(ri_j / np.std(ri_j))) - (
        entropy(xi_std) + entropy(rj_i / np.std(rj_i))
    )


'''def find_first_significant_jump(arr, q=0.4, k=5.0, min_gap_rel=0.02):
    """
    Returns the index of the first element AFTER the first significant jump.

    q           fraction of leading diffs used as 'flat head' to estimate noise
    k           how many MADs above noise counts as significant
    min_gap_rel minimum relative jump vs total range to avoid tiny absolute jumps
    """
    arr = np.asarray(arr, float)
    if arr.size < 3:
        return None  # not enough points to define a jump

    diffs = np.diff(arr)
    n = len(diffs)
    head = max(1, int(np.floor(q * n)))
    head_diffs = diffs[:head]

    # MAD as a robust noise scale
    med = np.median(head_diffs)
    noise = np.median(np.abs(head_diffs - med))

    abs_thresh = k * noise
    rel_thresh = min_gap_rel * (arr[-1] - arr[0])
    thresh = max(abs_thresh, rel_thresh)

    idxs = np.where(diffs >= thresh)[0]
    return int(idxs[0] + 1) if len(idxs) else None
'''
def find_significant_jump(arr):
    # Calculate ratios between consecutive elements
    ratios = arr[1:] / arr[:-1]
    print(f"ratios: {ratios}")
    # Find index of maximum ratio
    jump_index = np.argmax(ratios)
    return jump_index + 1  # +1 because ratios array is shifted

def find_significant_jump_robust(arr, epsilon=None):
    """
    Find the single most significant jump (largest ratio)
    """
    arr = np.array(arr)
    
    if len(arr) < 2:
        raise ValueError("Need at least 2 values to detect jumps")
    
    # Auto-calculate epsilon
    if epsilon is None:
        non_zero_vals = arr[arr > 0]
        if len(non_zero_vals) > 0:
            epsilon = np.min(non_zero_vals) * 1e-10
        else:
            epsilon = 1e-15
    
    arr_safe = arr + epsilon
    
    # Calculate log differences
    log_diffs = np.diff(np.log10(arr_safe))
    
    # Filter: only consider transitions where start value is meaningful
    valid_mask = arr[:-1] > epsilon * 10
    
    if not np.any(valid_mask):
        return 1  # Default to first transition
    
    # Find largest valid jump
    valid_log_diffs = log_diffs.copy()
    valid_log_diffs[~valid_mask] = -np.inf  # Ignore invalid jumps
    
    return np.argmax(valid_log_diffs) + 1
    
def search_causal_order_with_log(X, U):
    """Search the causal ordering for the current step.

    Parameters
    ----------
    X : np.ndarray
    U : array-like of remaining variable indices
    log_df : pd.DataFrame or None
        If provided, we will append the scores for this iteration.
    step_idx : int or None
        Current iteration number in the outer loop (0,1,2,...)
    """
    Uc, Vj = search_candidate(U)

    if len(Uc) == 1:
        return [], []

    M_list = []
    L_list = []
    for i in Uc:
        M = 0
        L = 0
        for j in U:
            if i != j:
                xi_std = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
                xj_std = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])
                ri_j = (
                    xi_std
                    if i in Vj and j in Uc
                    else compute_residual(xi_std, xj_std)
                )
                rj_i = (
                    xj_std
                    if j in Vj and i in Uc
                    else compute_residual(xj_std, xi_std)
                )
                M += np.min([0, diff_mutual_info(xi_std, xj_std, ri_j, rj_i)]) ** 2
                L += np.max([0, diff_mutual_info(xi_std, xj_std, ri_j, rj_i)]) ** 2

        M_list.append(-1.0 * M)
        L_list.append(L)
    
    M_values = np.abs(np.asarray(M_list, dtype=float))   # aligned with Uc
    L_values = np.asarray(L_list, dtype=float)   # aligned with Uc

    #print(f"M (roots score): {M_values}")
    #print(f"L (leaves score): {L_values}")

    # --- sort ascending and keep the permutation of indices (over Uc) ---
    M_sorted_pos = np.argsort(M_values)          # positions in Uc
    L_sorted_pos = np.argsort(L_values)          # positions in Uc
    #print(f"M_sorted_pos: {M_sorted_pos}")
    #print(f"L_sorted_pos: {L_sorted_pos}")

    M_sorted = M_values[M_sorted_pos]
    L_sorted = L_values[L_sorted_pos]
    print(M_sorted)
    #print(f"M_sorted: {M_sorted}")
    #print(f"L_sorted: {L_sorted}")
    # Elbows (last root / last leaf positions in the sorted arrays)
    last_root_pos  = find_significant_jump_robust(M_sorted)
    last_leaf_pos  = find_significant_jump_robust(L_sorted)

    # Select everything up to each elbow (inclusive), map back to indices in Uc
    #idx_roots  = Uc[M_sorted_pos[: last_root_pos + 1]]
    #idx_leaves = Uc[L_sorted_pos[: last_leaf_pos + 1]]
    #print(f"Uc: {Uc}")
    
    #print(f"idx_roots: {idx_roots}")
    #print(f"idx_leaves: {idx_leaves}")
    #print(last_root_pos)
    #print(last_leaf_pos)
    idx_roots = M_sorted_pos[:last_root_pos]
    idx_leaves = L_sorted_pos[:last_leaf_pos]

    return idx_roots, idx_leaves




def fit_with_log(X):
    """Fit the model to X and log each screening step."""
    X = check_array(X)
    n_features = X.shape[1]

    # variables 
    U = np.arange(n_features) 
    X_ = np.copy(X)

    # getting first and last level
    idx_roots, idx_leaves = search_causal_order_with_log(X_, U)
     
    
    return idx_roots, idx_leaves
