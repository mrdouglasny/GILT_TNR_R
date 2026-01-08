#!/usr/bin/env python3
"""
Test if higher chi stabilizes Z3 flow.
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

def test_chi(chi, n_steps=5):
    """Test Z3 flow at given chi."""
    print(f"\n{'='*60}")
    print(f"Testing χ = {chi}")
    print(f"{'='*60}")

    pars = {
        'q': 3,
        'relT': 1.0,
        'gilt_eps': 1e-6,
        'cg_chis': [chi],
        'cg_eps': 1e-10,
        'verbosity': 0,
        'symmetry_tensors': True,
    }

    A = get_initial_tensor_potts_relT(pars)
    log_fact = 0.0

    print(f"\n{'Step':>4} {'x_σ':>10} {'x_ε':>10} {'σ err%':>10} {'ε err%':>10}")
    print("-" * 50)

    results = []
    for step in range(n_steps + 1):
        sd = get_scaldims_potts(A)
        sigma_err = abs(sd[1] - X_SIGMA) / X_SIGMA * 100
        eps_err = abs(sd[3] - X_EPS) / X_EPS * 100
        print(f"{step:>4} {sd[1]:>10.4f} {sd[3]:>10.4f} {sigma_err:>9.1f}% {eps_err:>9.1f}%")
        results.append((sd[1], sd[3], sigma_err, eps_err))

        if step < n_steps:
            A, log_fact = gilttnr_step(A, log_fact, pars)

    return results

def main():
    print("=" * 60)
    print("Z3 Flow: Chi Dependence Study")
    print("=" * 60)
    print(f"CFT targets: x_σ = {X_SIGMA:.4f}, x_ε = {X_EPS:.4f}")

    chis = [12, 16, 20, 24]
    n_steps = 4

    all_results = {}
    for chi in chis:
        all_results[chi] = test_chi(chi, n_steps)

    # Summary table
    print("\n" + "=" * 60)
    print("Summary: Final scaling dimensions after", n_steps, "steps")
    print("=" * 60)
    print(f"\n{'χ':>6} {'x_σ':>10} {'x_ε':>10} {'σ err%':>10} {'ε err%':>10}")
    print("-" * 50)
    for chi in chis:
        final = all_results[chi][-1]
        print(f"{chi:>6} {final[0]:>10.4f} {final[1]:>10.4f} {final[2]:>9.1f}% {final[3]:>9.1f}%")

if __name__ == "__main__":
    main()
