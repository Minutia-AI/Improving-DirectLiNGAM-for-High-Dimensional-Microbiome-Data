from src.root_screening import fit_with_log
import pandas as pd
import numpy as np
from src.data_sim_ba import simulate_data
from lingam.utils import make_prior_knowledge
import lingam
from utils.experiments import invert_permutation, compute_metrics
from itertools import product
from joblib import Parallel, delayed
import time

# --- parameter grids ---
P_LIST = [100, 300, 500]
M_LIST = [1]
N_LIST = [2_000, 6_000, 10_000]
RUNS = 10

# --- single experiment ---
def run_experiment_roots_screening(p, m, n, seed):
    
    # simulate
    
    X, perm, W = simulate_data(p=p, m=m, n=n, seed=seed)
    
    # start the timer
    start_roots_screening = time.perf_counter()  
    
    # estimate roots in permuted X space
    roots_estimated, _ = fit_with_log(X)
    # perm_roots_est = perm[roots_estimated] to be used in case of root nodes accuracy assessments

    
    # execution of direct lingam with background knowledge.
    
    # keep in mind that roots_estimated must be used cause in this case we are working in the permuted space.
    prior_knowledge = make_prior_knowledge(
                                            n_variables=p,
                                            exogenous_variables=roots_estimated.tolist(),
                                            )

    # fit the model with prior knowledge
    model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
    model.fit(X)

    # execution time of the roots screening lingam
    elapsed_roots_screening = time.perf_counter() - start_roots_screening 
    
    # metrics roots screening

    est_adj = model.adjacency_matrix_
    # retrieving the correct matrix state
    est_adj = invert_permutation(est_adj, perm)
    mse_roots_screening, shd_roots_screening, prec_roots_screening, rec_roots_screening = compute_metrics(B_true=W, B_hat=est_adj)


    # fit the model without prior knowledge (plain direct lingam)
    
    # start the timer
    start_direct_lingam = time.perf_counter()  
    
    model = lingam.DirectLiNGAM()
    model.fit(X)

    # execution time of the plain DirectLingam
    elapsed_direct_lingam = time.perf_counter() - start_direct_lingam
    
    # metrics roots screening
    
    est_adj = model.adjacency_matrix_
    # retrieving the correct matrix state
    est_adj = invert_permutation(est_adj, perm)
    mse_direct_lingam, shd_direct_lingam, prec_direct_lingam, rec_direct_lingam = compute_metrics(B_true=W, B_hat=est_adj)


    return {
        "precision_roots_screening": prec_roots_screening,
        "recall_roots_screening":rec_roots_screening,
        "mse_roots_screening": mse_roots_screening,
        "shd_roots_screening": shd_roots_screening,
        "precision_direct_lingam": prec_direct_lingam,
        "recall_direct_lingam":rec_direct_lingam,
        "mse_direct_lingam": mse_direct_lingam,
        "shd_direct_lingam": shd_direct_lingam,
        "p": p,
        "m": m,
        "n": n,
        #"root_frac": root_frac,
        "random_state": seed,
        #"true_roots": list(roots_true_orig),
        #"estimated_roots": list(perm[roots_estimated]),
        "exec_time_sec_roots_screening": elapsed_roots_screening,
        "exec_time_sec_direct_lingam": elapsed_direct_lingam
    }

def main():
    # --- job list ---
    jobs = list(product(P_LIST, M_LIST, N_LIST, range(RUNS)))

    # --- run in parallel ---
    results_list = Parallel(
        n_jobs=-1,          # all cores; set to an int to limit
        backend="loky",     # process-based (default); good for CPU-bound
        verbose=10,         # nice progress logs
        batch_size="auto",  # chunking for efficiency
    )(
        delayed(run_experiment_roots_screening)(p, m, n, seed)
        for (p, m, n, seed) in jobs
    )

    # --- assemble & save ---
    df_results = pd.DataFrame(results_list)
    df_results.to_csv("execution_time_comparison/root_screening_lingam_ba_network.csv", index=False)
    print("Saved /execution_time_comparison/root_screening_lingam_ba_network.csv")

if __name__ == "__main__":
    main()