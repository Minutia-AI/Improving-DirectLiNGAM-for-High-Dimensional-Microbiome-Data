import numpy as np
from cdt.metrics import SHD 

def compute_metrics(B_true, B_hat, threshold=0.0):
    """
    Compute SHD, precision, and recall using CDT's built-in metrics.
    Arguments:
        B_true, B_hat : 2D arrays (adjacency matrices)
        threshold     : values below this are considered 0 (to handle float noise)
    Returns:
        mse, shd, precision, recall
    """
    # Binarize adjacency matrices
    B_true_bin = (np.abs(B_true) > threshold).astype(int)
    B_hat_bin  = (np.abs(B_hat)  > threshold).astype(int)

    TP = np.sum((B_true_bin == 1) & (B_hat_bin == 1))
    FP = np.sum((B_true_bin == 0) & (B_hat_bin == 1))
    FN = np.sum((B_true_bin == 1) & (B_hat_bin == 0))
    precision = TP / (TP + FP + 1e-12)
    recall    = TP / (TP + FN + 1e-12)

    # CDT SHD (Structural Hamming Distance)
    shd = SHD(B_true_bin, B_hat_bin)

    # MSE (parameter-level)
    mse = float(np.mean((B_true - B_hat) ** 2))

    return mse, shd, precision, recall

def invert_permutation(B, perm):
    inverse_perm = np.argsort(perm)
    B_perm_inverted = B[inverse_perm, :][:, inverse_perm]
    return B_perm_inverted