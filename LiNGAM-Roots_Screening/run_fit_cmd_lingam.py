import lingam
import pandas as pd
import time
import numpy as np


df_cmd = pd.read_csv("./real_data_application/exp4/data/clr_transformation_cmd_healthy_no_standardization_30percent_presence.csv")
p = len(df_cmd.columns)
feature_names = df_cmd.columns.to_list()
X = df_cmd.values
start_roots_screening = time.perf_counter()  

# fit the model with prior knowledge
model = lingam.DirectLiNGAM()
model.fit(X)

# execution time of the roots screening lingam
elapsed_roots_screening = time.perf_counter() - start_roots_screening 

print(f"elapsed_roots_screening = {elapsed_roots_screening:.3f} s")

A = model.adjacency_matrix_  # A[i, j] = effect of j -> i


# 1) Save adjacency + feature names efficiently (compressed binary)
np.savez_compressed(
    "adjacency_model_run_time_lingam.npz",
    adjacency=A.astype(np.float32),
    features=np.array(feature_names, dtype=object),
    elapsed_seconds=np.array([elapsed_roots_screening], dtype=np.float64),
    )
