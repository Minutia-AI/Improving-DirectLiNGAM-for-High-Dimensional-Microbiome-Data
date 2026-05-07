from itertools import product
from joblib import Parallel, delayed
import pandas as pd
from utils.experiments import run_experiment
import os

# --- helper functions ---
def s_from_k(p, k): 
    return float(k)/float(max(1, p-1))

def n_from_ratio(p, ratio): 
    return max(10, int(round(ratio*p)))


def wrapper_call(p, k, ratio, n_models):
    s = s_from_k(p, k)
    n = n_from_ratio(p, ratio)
    return run_experiment(p=p, n=n, s=s, n_models=n_models)

# --- parameter grids ---
P = [20]      # number of variables
K = [2,3]            # expected parents per node
RATIOS = [3.0]   # n = ratio * p
N_MODELS = 100

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "experiments", "results.csv")

# --- parallel execution ---
results_blocks = Parallel(n_jobs=-1, verbose=10)(
    delayed(wrapper_call)(p, k, ratio, N_MODELS)
    for p, k, ratio in product(P, K, RATIOS)
)

# flatten (each block = list of dicts)
results_list = [row for block in results_blocks for row in block]

# --- save results ---
results_df = pd.DataFrame(results_list)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {len(results_df)} rows to {OUTPUT_PATH}")