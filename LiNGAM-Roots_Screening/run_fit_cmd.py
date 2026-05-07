from src.root_screening import fit_with_log
from lingam.utils import make_prior_knowledge
import lingam
import pandas as pd
import time
import numpy as np


df_cmd = pd.read_csv("./real_data_application/exp4/data/clr_transformation_cmd_healthy_no_standardization_30percent_presence.csv")
p = len(df_cmd.columns)
print(p)
feature_names = df_cmd.columns.to_list()
X = df_cmd.values
start_roots_screening = time.perf_counter()  
roots_estimated, _ = fit_with_log(X)
print(roots_estimated)
prior_knowledge = make_prior_knowledge(
                                            n_variables=p,
                                            exogenous_variables=roots_estimated.tolist(),
                                            )

# fit the model with prior knowledge
model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
model.fit(X)

# execution time of the roots screening lingam
elapsed_roots_screening = time.perf_counter() - start_roots_screening 

print(f"elapsed_roots_screening = {elapsed_roots_screening:.3f} s")

A = model.adjacency_matrix_  # A[i, j] = effect of j -> i


# 1) Save adjacency + feature names efficiently (compressed binary)
np.savez_compressed(
    "adjacency_model_run_time_roots_screening_30_percent.npz",
    adjacency=A.astype(np.float32),
    features=np.array(feature_names, dtype=object),
    elapsed_seconds=np.array([elapsed_roots_screening], dtype=np.float64),
    )
