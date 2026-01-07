#!/usr/bin/env python3
"""
Test Gilt-TNR flow with TensorZ3 for 3-state Potts model.

This verifies that:
1. TensorZ3 and plain Tensor give identical results
2. The flow is stable near the critical point
3. Scaling dimensions converge to CFT values
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from GiltTNR2D import gilttnr_step
from GiltTNR2D_Potts import (
    get_initial_tensor_potts_relT,
    get_scaldims_potts,
    POTTS3_CFT_DIMENSIONS
)

def main():
    print("=" * 70)
    print("3-State Potts Model: TensorZ3 vs Plain Tensor Flow Test")
    print("=" * 70)

    # CFT values
    x_sigma = POTTS3_CFT_DIMENSIONS['spin']  # 2/15 ≈ 0.1333
    x_eps = POTTS3_CFT_DIMENSIONS['energy']  # 4/5 = 0.8

    # Parameters
    chi = 16
    n_steps = 7
    relT = 1.0  # At critical point

    pars = {
        'q': 3,
        'relT': relT,
        'gilt_eps': 1e-6,  # Enable Gilt
        'cg_chis': [chi],
        'cg_eps': 1e-10,
        'verbosity': 0,
    }

    print(f"\nParameters: χ={chi}, n_steps={n_steps}, relT={relT}")
    print(f"CFT targets: x_σ = {x_sigma:.4f}, x_ε = {x_eps:.4f}")
    print("-" * 70)

    # Test 1: Plain Tensor
    print("\n1. Plain Tensor flow:")
    print(f"{'Step':>4} {'x_1':>10} {'x_2':>10} {'x_3':>10} {'x_ε err':>10}")
    print("-" * 50)

    pars_plain = dict(pars)
    pars_plain['symmetry_tensors'] = False
    A_plain = get_initial_tensor_potts_relT(pars_plain)
    log_fact_plain = 0.0

    for step in range(n_steps):
        sd = get_scaldims_potts(A_plain)
        x_eps_err = abs(sd[3] - x_eps) / x_eps * 100 if len(sd) > 3 else float('nan')
        print(f"{step:>4} {sd[1]:>10.4f} {sd[2]:>10.4f} {sd[3]:>10.4f} {x_eps_err:>9.1f}%")

        A_plain, log_fact_plain = gilttnr_step(A_plain, log_fact_plain, pars_plain)

    # Final scaling dimensions
    sd_plain = get_scaldims_potts(A_plain)

    # Test 2: TensorZ3
    print("\n2. TensorZ3 flow:")
    print(f"{'Step':>4} {'x_1':>10} {'x_2':>10} {'x_3':>10} {'x_ε err':>10}")
    print("-" * 50)

    pars_z3 = dict(pars)
    pars_z3['symmetry_tensors'] = True
    A_z3 = get_initial_tensor_potts_relT(pars_z3)
    log_fact_z3 = 0.0

    for step in range(n_steps):
        sd = get_scaldims_potts(A_z3)
        x_eps_err = abs(sd[3] - x_eps) / x_eps * 100 if len(sd) > 3 else float('nan')
        print(f"{step:>4} {sd[1]:>10.4f} {sd[2]:>10.4f} {sd[3]:>10.4f} {x_eps_err:>9.1f}%")

        A_z3, log_fact_z3 = gilttnr_step(A_z3, log_fact_z3, pars_z3)

    # Final scaling dimensions
    sd_z3 = get_scaldims_potts(A_z3)

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"\nPlain Tensor final: x_σ = {sd_plain[1]:.4f}, x_ε = {sd_plain[3]:.4f}")
    print(f"TensorZ3 final:     x_σ = {sd_z3[1]:.4f}, x_ε = {sd_z3[3]:.4f}")
    print(f"CFT exact:          x_σ = {x_sigma:.4f}, x_ε = {x_eps:.4f}")

    print(f"\nPlain error: x_σ: {abs(sd_plain[1] - x_sigma)/x_sigma*100:.1f}%, x_ε: {abs(sd_plain[3] - x_eps)/x_eps*100:.1f}%")
    print(f"Z3 error:    x_σ: {abs(sd_z3[1] - x_sigma)/x_sigma*100:.1f}%, x_ε: {abs(sd_z3[3] - x_eps)/x_eps*100:.1f}%")

    # Check if results match
    match = np.allclose(sd_plain[:4], sd_z3[:4], rtol=0.01)
    print(f"\nPlain and Z3 match (1% tolerance): {'✓ YES' if match else '✗ NO'}")

if __name__ == "__main__":
    main()
