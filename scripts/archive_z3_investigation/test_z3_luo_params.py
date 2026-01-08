#!/usr/bin/env python3
"""
Test Z3 flow with parameters from Luo et al. (arXiv:2408.10312):
- χ = 30
- gilt_eps = 3e-5 (larger than our 1e-6)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from tensors import Tensor, TensorZ3
from GiltTNR2D import gilttnr_step
from GiltTNR2D_Potts import get_initial_tensor_potts_relT, get_scaldims_potts

# CFT values
X_SIGMA = 2/15  # 0.1333
X_EPS = 4/5     # 0.8

def test_flow(chi, gilt_eps, use_symmetry, n_steps=5, label=""):
    """Test flow with given parameters."""
    print(f"\n{'='*60}")
    print(f"{label}: χ={chi}, gilt_eps={gilt_eps:.0e}, symmetry={use_symmetry}")
    print(f"{'='*60}")

    pars = {
        'q': 3,
        'relT': 1.0,
        'gilt_eps': gilt_eps,
        'cg_chis': [chi],
        'cg_eps': 1e-10,
        'verbosity': 0,
        'symmetry_tensors': use_symmetry,
    }

    A = get_initial_tensor_potts_relT(pars)
    log_fact = 0.0

    print(f"\n{'Step':>4} {'x_σ':>10} {'x_ε':>10} {'σ err%':>10} {'ε err%':>10}")
    print("-" * 50)

    for step in range(n_steps + 1):
        sd = get_scaldims_potts(A)
        sigma_err = abs(sd[1] - X_SIGMA) / X_SIGMA * 100
        eps_err = abs(sd[3] - X_EPS) / X_EPS * 100
        print(f"{step:>4} {sd[1]:>10.4f} {sd[3]:>10.4f} {sigma_err:>9.1f}% {eps_err:>9.1f}%")

        if step < n_steps:
            A, log_fact = gilttnr_step(A, log_fact, pars)

    return sd[1], sd[3]

def main():
    print("=" * 60)
    print("Testing Luo et al. parameters (arXiv:2408.10312)")
    print("=" * 60)
    print(f"CFT targets: x_σ = {X_SIGMA:.4f}, x_ε = {X_EPS:.4f}")

    n_steps = 5

    # Test 1: Our original parameters (Plain)
    test_flow(16, 1e-6, False, n_steps, "Plain χ=16, eps=1e-6")

    # Test 2: Our original parameters (Z3)
    test_flow(16, 1e-6, True, n_steps, "Z3 χ=16, eps=1e-6")

    # Test 3: Luo et al. parameters (Plain)
    test_flow(30, 3e-5, False, n_steps, "Plain χ=30, eps=3e-5 (Luo)")

    # Test 4: Luo et al. parameters (Z3)
    test_flow(30, 3e-5, True, n_steps, "Z3 χ=30, eps=3e-5 (Luo)")

    # Test 5: Higher gilt_eps only
    test_flow(16, 3e-5, True, n_steps, "Z3 χ=16, eps=3e-5")

    # Test 6: Try even higher gilt_eps
    test_flow(16, 1e-4, True, n_steps, "Z3 χ=16, eps=1e-4")

if __name__ == "__main__":
    main()
