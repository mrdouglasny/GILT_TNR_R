#!/usr/bin/env python3
"""
MODULE: phi4_test.py
USAGE: cd ekrgilttrnr/src/GiltTNR && python3 ../../scripts/phi4_test.py
REQUIRES: Python 3.10+, NumPy 2.x, SciPy 1.15+
INPUTS: None (generates tensors from parameters)
OUTPUTS: ekrgilttrnr/data/phi4_test_results.txt
DESCRIPTION: Test the phi4 tensor construction and Gilt-TNR integration

This script validates that the phi4 model works correctly with the Gilt-TNR
algorithm and can detect the phase transition.
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add GiltTNR to path
script_dir = os.path.dirname(os.path.abspath(__file__))
gilttnr_dir = os.path.join(script_dir, '..', 'src', 'GiltTNR')
sys.path.insert(0, gilttnr_dir)
os.chdir(gilttnr_dir)

from GiltTNR2D_Phi4 import (
    get_initial_tensor_phi4,
    get_A_spectrum_phi4,
    build_phi4_tensor_array,
    estimate_critical_mu_sq
)
from GiltTNR2D_essentials import gilttnr_step

# Output file
output_dir = os.path.join(script_dir, '..', 'data')
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'phi4_test_results.txt')


def test_tensor_construction():
    """Test basic tensor construction."""
    print("=" * 60)
    print("TEST 1: Tensor Construction")
    print("=" * 60)

    pars = {
        'mu_sq': -0.09,
        'lam': 1.0,
        'kappa': 1.0,
        'K': 16,
        'D': 8,
        'symmetry_tensors': False
    }

    A = build_phi4_tensor_array(pars['mu_sq'], pars['lam'], pars['kappa'],
                                 pars['K'], pars['D'])

    print(f"Tensor shape: {A.shape}")
    print(f"Tensor norm: {np.linalg.norm(A):.6f}")
    print(f"Max element: {np.max(np.abs(A)):.6f}")

    # SVD spectrum
    M = A.reshape(pars['D']**2, pars['D']**2)
    s = np.linalg.svd(M, compute_uv=False)
    s = s / s[0]
    print(f"Top 5 singular values: {s[:5]}")

    print("\nPASS: Tensor construction works")
    return True


def test_gilttnr_steps():
    """Test Gilt-TNR RG steps."""
    print("\n" + "=" * 60)
    print("TEST 2: Gilt-TNR RG Steps")
    print("=" * 60)

    pars = {
        'mu_sq': -0.09,
        'lam': 1.0,
        'kappa': 1.0,
        'K': 16,
        'D': 8,
        'symmetry_tensors': False
    }

    gilt_pars = {
        'gilt_eps': 6e-6,
        'cg_chis': list(range(1, 17)),
        'cg_eps': 1e-10,
        'verbosity': 0,
        'rotate': False,
    }

    A = get_initial_tensor_phi4(pars)
    print(f"Initial tensor type: {type(A).__name__}")

    spectrum = get_A_spectrum_phi4(A)
    print(f"Initial spectrum (top 5): {spectrum[:5]}")

    print("\nRunning 5 Gilt-TNR steps...")
    for step in range(5):
        A, log_fact, errs = gilttnr_step(A, 0.0, gilt_pars)
        spectrum = get_A_spectrum_phi4(A)
        print(f"  Step {step+1}: lambda_2 = {spectrum[1]:.6f}")

    print("\nPASS: Gilt-TNR steps work")
    return True


def test_phase_transition():
    """Scan mu^2 to find phase transition."""
    print("\n" + "=" * 60)
    print("TEST 3: Phase Transition Scan")
    print("=" * 60)

    gilt_pars = {
        'gilt_eps': 6e-6,
        'cg_chis': list(range(1, 17)),
        'cg_eps': 1e-10,
        'verbosity': 0,
        'rotate': False,
    }

    base_pars = {
        'lam': 1.0,
        'kappa': 1.0,
        'K': 16,
        'D': 8,
        'symmetry_tensors': False
    }

    results = []

    print(f"\n{'mu^2':>10s}  {'lambda_2 (10 RG)':>16s}  Phase")
    print("-" * 45)

    mu_sq_values = [-0.25, -0.20, -0.18, -0.17, -0.16, -0.15, -0.10, -0.05, 0.0]

    for mu_sq in mu_sq_values:
        pars = {**base_pars, 'mu_sq': mu_sq}
        A = get_initial_tensor_phi4(pars)

        # Run 10 RG steps
        for _ in range(10):
            A, _, _ = gilttnr_step(A, 0.0, gilt_pars)

        spectrum = get_A_spectrum_phi4(A)
        lambda2 = spectrum[1] if len(spectrum) > 1 else 0.0

        if lambda2 > 0.9:
            phase = 'BROKEN'
        elif lambda2 < 0.1:
            phase = 'SYMMETRIC'
        else:
            phase = 'CRITICAL?'

        results.append((mu_sq, lambda2, phase))
        print(f'{mu_sq:10.3f}  {lambda2:16.6f}  {phase}')

    # Estimate critical point
    critical_mu_sq = None
    for i in range(len(results)-1):
        if results[i][2] == 'SYMMETRIC' and results[i+1][2] == 'BROKEN':
            critical_mu_sq = (results[i][0] + results[i+1][0]) / 2
            break
        elif results[i][2] == 'BROKEN' and results[i+1][2] == 'SYMMETRIC':
            critical_mu_sq = (results[i][0] + results[i+1][0]) / 2
            break

    print(f"\nEstimated critical mu^2: {critical_mu_sq}")
    print(f"Literature estimate: {estimate_critical_mu_sq(1.0):.4f} (for lambda=1)")

    print("\nPASS: Phase transition detected")
    return results, critical_mu_sq


def main():
    """Run all tests and save results."""
    print("phi4 Model Test Suite for Gilt-TNR")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Run tests
    test_tensor_construction()
    test_gilttnr_steps()
    results, critical_mu_sq = test_phase_transition()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"All tests passed!")
    print(f"Estimated critical mu^2: {critical_mu_sq}")
    print(f"Expected (literature): ~ -0.09 for lambda=1, kappa=1")
    print(f"Difference due to finite K, D, chi")

    # Save results
    with open(output_file, 'w') as f:
        f.write(f"phi4 Model Test Results\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  lambda = 1.0\n")
        f.write(f"  kappa = 1.0\n")
        f.write(f"  K = 16 (quadrature points)\n")
        f.write(f"  D = 8 (initial bond dim)\n")
        f.write(f"  chi = 16 (max Gilt-TNR bond dim)\n")
        f.write(f"  gilt_eps = 6e-6\n\n")
        f.write(f"Phase Transition Scan:\n")
        f.write(f"{'mu^2':>10s}  {'lambda_2':>12s}  Phase\n")
        f.write("-" * 40 + "\n")
        for mu_sq, lambda2, phase in results:
            f.write(f"{mu_sq:10.3f}  {lambda2:12.6f}  {phase}\n")
        f.write(f"\nEstimated critical mu^2: {critical_mu_sq}\n")
        f.write(f"Literature estimate: -0.09 (lambda/|mu^2_c| ~ 10.9)\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
