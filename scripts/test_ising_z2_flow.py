#!/usr/bin/env python3
"""
Test Gilt-TNR flow with TensorZ2 for Ising model.
Compare with Potts TensorZ3 to understand why Z2 works but Z3 doesn't.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from GiltTNR2D import gilttnr_step
from GiltTNR2D_Ising_benchmarks import get_initial_tensor, get_scaldims

def main():
    print("=" * 70)
    print("Ising Model: TensorZ2 vs Plain Tensor Flow Test")
    print("=" * 70)

    # CFT values
    x_sigma = 1/8  # 0.125
    x_eps = 1.0

    # Parameters
    chi = 16
    n_steps = 7
    beta = np.log(1 + np.sqrt(2)) / 2  # Critical point

    pars_base = {
        'beta': beta,
        'gilt_eps': 1e-6,
        'cg_chis': [chi],
        'cg_eps': 1e-10,
        'verbosity': 0,
    }

    print(f"\nParameters: χ={chi}, n_steps={n_steps}, β={beta:.6f}")
    print(f"CFT targets: x_σ = {x_sigma:.4f}, x_ε = {x_eps:.4f}")
    print("-" * 70)

    # Test 1: Plain Tensor
    print("\n1. Plain Tensor flow:")
    print(f"{'Step':>4} {'x_1':>10} {'x_2':>10} {'x_3':>10}")
    print("-" * 40)

    pars_plain = dict(pars_base)
    pars_plain['symmetry_tensors'] = False
    A_plain = get_initial_tensor(pars_plain)
    log_fact_plain = 0.0

    for step in range(n_steps):
        sd = get_scaldims(A_plain, pars_plain)
        print(f"{step:>4} {sd[1]:>10.4f} {sd[2]:>10.4f} {sd[3] if len(sd) > 3 else float('nan'):>10.4f}")
        A_plain, log_fact_plain = gilttnr_step(A_plain, log_fact_plain, pars_plain)

    sd_plain = get_scaldims(A_plain, pars_plain)

    # Test 2: TensorZ2
    print("\n2. TensorZ2 flow:")
    print(f"{'Step':>4} {'x_1':>10} {'x_2':>10} {'x_3':>10}")
    print("-" * 40)

    pars_z2 = dict(pars_base)
    pars_z2['symmetry_tensors'] = True
    A_z2 = get_initial_tensor(pars_z2)
    log_fact_z2 = 0.0

    print(f"   Initial tensor type: {type(A_z2)}")

    for step in range(n_steps):
        sd = get_scaldims(A_z2, pars_z2)
        print(f"{step:>4} {sd[1]:>10.4f} {sd[2]:>10.4f} {sd[3] if len(sd) > 3 else float('nan'):>10.4f}")
        A_z2, log_fact_z2 = gilttnr_step(A_z2, log_fact_z2, pars_z2)

    sd_z2 = get_scaldims(A_z2, pars_z2)

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"\nPlain Tensor final: x_σ = {sd_plain[1]:.4f}")
    print(f"TensorZ2 final:     x_σ = {sd_z2[1]:.4f}")
    print(f"CFT exact:          x_σ = {x_sigma:.4f}")

    # Check if results match
    match = np.allclose(sd_plain[:3], sd_z2[:3], rtol=0.01)
    print(f"\nPlain and Z2 match (1% tolerance): {'✓ YES' if match else '✗ NO'}")

if __name__ == "__main__":
    main()
