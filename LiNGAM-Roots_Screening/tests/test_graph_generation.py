import os
import sys
import numpy as np
import pytest
import lingam
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from data_sim import generate_triangular_mask_exact, simulate_data, invert_permutation


CASES = [
    (10, 3, 3),
    (10, 3, 1),
    (10, 1, 3)
]

@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("p,r,l", CASES)
def test_root_and_leaf_constraints(seed, p, r, l):
    
    rng = np.random.default_rng(seed)
    s = 0.1
    mask, roots, leaves = generate_triangular_mask_exact(
        p=p, s=s, r=r, l=l, rng=rng
    )
    # --- ROOTS: no incoming edges ---
    assert(mask[roots, :].sum(axis=1) == 0).all()
    # --- LEAVES: no outgoing edges ---
    assert (mask[:, leaves].sum(axis=0) == 0).all()
   

def test_data_generation_process():
    random_state = 1
    p = 10
    s = 0.10
    n = 10_000 
    root_frac = 0.30
    leaf_frac = 0.30

    X, perm, W, _, _ = simulate_data(p = p, s = s, n = n, root_frac = root_frac, leaf_frac = leaf_frac, seed=random_state )

    model = lingam.DirectLiNGAM()
    model.fit(X)
   
    est_adj = model.adjacency_matrix_
    est_adj = invert_permutation(est_adj, perm)

    # tolerance on weights
    alpha = 0.02    
    
    # boolean mask of true edges
    true_edges = (W != 0)  
    est_edges  = (np.abs(est_adj) > 0)
   

    # All true edges weights must be within alpha
    assert np.allclose(W, est_adj, atol=alpha)

    # The estimated structure must be the same (with enough samples)
    assert np.array_equal(true_edges, est_edges)