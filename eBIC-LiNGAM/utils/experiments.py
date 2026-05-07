from cdt.metrics import SHD 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import lars_path
from scipy.special import gammaln
import lingam
import pandas as pd
from utils.data_sim import simulate_data
from utils.data_sim import invert_permutation


def predict_adaptive_ebic_lasso(
    X, predictors, target,
    gamma=1.0,             # adaptive-lasso exponent (same as before)
    gamma_ebic=0.5,       # NEW: EBIC gamma (0.35–0.5 recommended)
    eps=1e-6               # numeric safety for zero pilot coefs
):
    """
    Node-wise adaptive lasso with EBIC selection (nEBIC-LiNGAM).
    - Pilot OLS -> adaptive weights (|beta_init|^gamma)
    - LARS path on reweighted design
    - Pick support by minimizing EBIC across path steps
    - OLS de-bias on original (unstandardized) X
    """
    # 1) Standardize once (so pilot magnitudes are comparable)
    X_std = StandardScaler().fit_transform(X)
    y_std = X_std[:, target]
    Xp_std = X_std[:, predictors]

    # 2) Pilot (OLS) -> adaptive weights
    lr = LinearRegression()
    lr.fit(Xp_std, y_std)
    weight = np.power(np.abs(lr.coef_) + eps, gamma)  # avoid zeros

    # 3) LARS path on reweighted predictors (equivalent to adaptive lasso)
    Xw = Xp_std * weight
    alphas, active, coef_path = lars_path(Xw, y_std, method="lasso")  # coef_path: (n_pred, n_steps)

    # 4) EBIC scan along the path
    n = Xw.shape[0]
    p_loc = Xw.shape[1]  # number of candidate parents for THIS node

    def ebic_of_support(support_mask):
        k = int(support_mask.sum())
        if k == 0:
            # When k = 0, the model predicts nothing → y_hat = 0.
            # Residuals are therefore equal to y itself.
            # RSS (Residual Sum of Squares) = y_stdᵀ*y_std = sum(y_i²)
            rss = float(y_std @ y_std)
            return n * np.log(rss / n + 1e-12)  # EBIC reduces to null-model fit
        # Extract only the active predictor columns (those with nonzero coefficients)
        Xs = Xw[:, support_mask]
        # Compute the ordinary least squares (OLS) solution on the selected predictors
        beta_ls, *_ = np.linalg.lstsq(Xs, y_std, rcond=None)
        # Compute residuals r = y_std - X_S β̂_S
        resid = y_std - Xs @ beta_ls
        # Compute Residual Sum of Squares (RSS = rᵀr = Σ r_i²)
        rss = float(resid @ resid)
        # Compute BIC
        bic = n * np.log(rss / n + 1e-12) + k * np.log(n)

        # Combinatorial correction (only if 0 < k < p_loc)
        # For k=0 (null model) or k=ploc (full model) there is exactly one subset, so (p0)=(pp)=1
        if 0 < k < p_loc:
            # log(p  k) = log(p!)−log(k!)−log((p−k)!)
            # to avoid numerical problem it can be rewrite using gammaln function
            # gammaln(p+1)−gammaln(k+1)−gammaln(p−k+1).
            logC = float(gammaln(p_loc + 1) - gammaln(k + 1) - gammaln(p_loc - k + 1))
            return bic + 2.0 * gamma_ebic * logC
        else:
            return bic

    best_score = np.inf
    best_mask = np.zeros(p_loc, dtype=bool)
    if coef_path.shape[1] == 0:
        # path is empty -> return all zeros
        pass
    else:
        for t in range(coef_path.shape[1]):
            mask = np.abs(coef_path[:, t]) > 0.0
            score = ebic_of_support(mask)
            if score < best_score:
                best_score = score
                best_mask = mask

    pruned_idx = best_mask  # boolean mask over predictors

    # 5) OLS de-bias on the original (unstandardized) X for selected parents
    coef = np.zeros(p_loc, dtype=float)
    if pruned_idx.any():
        pred = np.asarray(predictors)
        lr = LinearRegression()
        lr.fit(X[:, pred[pruned_idx]], X[:, target])
        coef[pruned_idx] = lr.coef_

    return coef


def estimate_adjacency_matrix_ebic(X, causal_order):
    """Estimate adjacency matrix by causal order using ebic approach
    Returns
    -------
    self : object
        Returns the adjacency matrix
    """
    adjacency_matrix = None
    B = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
    for i in range(1, len(causal_order)):
        target = causal_order[i]
        predictors = causal_order[:i]

        # target is exogenous variables if predictors are empty
        if len(predictors) == 0:
            continue

        B[target, predictors] = predict_adaptive_ebic_lasso(X, predictors, target)

    adjacency_matrix = B
    return adjacency_matrix



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

def run_experiment(p=20, n=5000, s=0.15, n_models = 30):
    results = []
    
    # iterate over models
    for m in range(n_models):
        # generating one dataset
        X, perm, B_true = simulate_data(p=p, s=s, n=n, seed=m)
        # baseline: DirectLiNGAM (BIC inside)
        model_BIC = lingam.DirectLiNGAM()
        model_BIC.fit(X)
        causal_order = model_BIC.causal_order_
        est_adj_BIC = model_BIC.adjacency_matrix_
        # retrieving the correct matrix state
        est_adj_BIC = invert_permutation(est_adj_BIC, perm)
        mse_bic, shd_bic, prec_bic, rec_bic = compute_metrics(B_true=B_true, B_hat=est_adj_BIC)
        # saving the results
        results.append({
            "run": m,
            "method": "BIC",
            "p": p, "n": n, "s": s, "seed": m,
            "mse": mse_bic,
            "shd": shd_bic,
            "precision": prec_bic,
            "recall": rec_bic
        })


        # LiNGAM with BIC
        est_adj_EBIC = estimate_adjacency_matrix_ebic(X = X, causal_order = causal_order)
        # retrieving the correct matrix state
        est_adj_EBIC = invert_permutation(est_adj_EBIC, perm)
        
        mse_ebic, shd_ebic, prec_ebic, rec_ebic = compute_metrics(B_true=B_true, B_hat=est_adj_EBIC)
        # saving the results
        results.append({
            "run": m,
            "method": "EBIC",
            "p": p, "n": n, "s": s, "seed": m,
            "mse": mse_ebic,
            "shd": shd_ebic,
            "precision": prec_ebic,
            "recall": rec_ebic
        })
    # save the results in a csv     
    # df = pd.DataFrame(results)
    # df.to_csv(csv_path, index=False)

    return results
        
        



