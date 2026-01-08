#!/usr/bin/env python3
"""
Test balanced_sectors fix for TensorZ3 in Gilt-TNR flow.

Compares:
1. Plain tensor (reference)
2. TensorZ3 with greedy allocation (original bug)
3. TensorZ3 with balanced allocation (fix)
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

def run_flow(pars, n_steps, label):
    """Run Gilt-TNR flow and track scaling dimensions."""
    A = get_initial_tensor_potts_relT(pars)
    log_fact = 0.0

    results = []
    for step in range(n_steps + 1):
        sd = get_scaldims_potts(A)
        results.append({
            'step': step,
            'x_sigma': sd[1] if len(sd) > 1 else np.nan,
            'x_eps': sd[3] if len(sd) > 3 else np.nan,
            'shape': A.shape if hasattr(A, 'shape') else A.to_ndarray().shape,
        })

        if step < n_steps:
            try:
                A, log_fact = gilttnr_step(A, log_fact, pars)
            except Exception as e:
                print(f"  {label} step {step+1} failed: {e}")
                break

    return results


def main():
    print("=" * 70)
    print("3-State Potts: Plain vs TensorZ3 (Greedy) vs TensorZ3 (Balanced)")
    print("=" * 70)

    # CFT targets
    x_sigma_exact = POTTS3_CFT_DIMENSIONS['spin']  # 2/15 ≈ 0.1333
    x_eps_exact = POTTS3_CFT_DIMENSIONS['energy']  # 4/5 = 0.8

    chi = 16
    n_steps = 6

    base_pars = {
        'q': 3,
        'relT': 1.0,  # Critical point
        'gilt_eps': 1e-6,
        'cg_chis': [chi],
        'cg_eps': 1e-10,
        'verbosity': 0,
    }

    print(f"\nParameters: χ={chi}, n_steps={n_steps}")
    print(f"CFT targets: x_σ = {x_sigma_exact:.4f}, x_ε = {x_eps_exact:.4f}")

    # 1. Plain tensor (reference)
    print("\n" + "-" * 70)
    print("1. Plain Tensor (reference)")
    print("-" * 70)
    pars_plain = dict(base_pars)
    pars_plain['symmetry_tensors'] = False
    results_plain = run_flow(pars_plain, n_steps, "Plain")

    print(f"{'Step':>4} {'x_σ':>10} {'x_ε':>10}")
    for r in results_plain:
        print(f"{r['step']:>4} {r['x_sigma']:>10.4f} {r['x_eps']:>10.4f}")

    # 2. TensorZ3 with greedy (original)
    print("\n" + "-" * 70)
    print("2. TensorZ3 Greedy (original algorithm)")
    print("-" * 70)
    pars_z3_greedy = dict(base_pars)
    pars_z3_greedy['symmetry_tensors'] = True
    pars_z3_greedy['balanced_sectors'] = False  # Explicit: original algorithm
    results_z3_greedy = run_flow(pars_z3_greedy, n_steps, "Z3-Greedy")

    print(f"{'Step':>4} {'x_σ':>10} {'x_ε':>10}")
    for r in results_z3_greedy:
        print(f"{r['step']:>4} {r['x_sigma']:>10.4f} {r['x_eps']:>10.4f}")

    # 3. TensorZ3 with balanced (fix)
    print("\n" + "-" * 70)
    print("3. TensorZ3 Balanced (with fix)")
    print("-" * 70)
    pars_z3_balanced = dict(base_pars)
    pars_z3_balanced['symmetry_tensors'] = True
    pars_z3_balanced['balanced_sectors'] = True  # Use the fix
    results_z3_balanced = run_flow(pars_z3_balanced, n_steps, "Z3-Balanced")

    print(f"{'Step':>4} {'x_σ':>10} {'x_ε':>10}")
    for r in results_z3_balanced:
        print(f"{r['step']:>4} {r['x_sigma']:>10.4f} {r['x_eps']:>10.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if len(results_plain) > 0 and len(results_z3_greedy) > 0 and len(results_z3_balanced) > 0:
        final_plain = results_plain[-1]
        final_z3_greedy = results_z3_greedy[-1]
        final_z3_balanced = results_z3_balanced[-1]

        print(f"\nAfter {final_plain['step']} steps:")
        print(f"  Plain:       x_σ = {final_plain['x_sigma']:.4f}, x_ε = {final_plain['x_eps']:.4f}")
        print(f"  Z3 Greedy:   x_σ = {final_z3_greedy['x_sigma']:.4f}, x_ε = {final_z3_greedy['x_eps']:.4f}")
        print(f"  Z3 Balanced: x_σ = {final_z3_balanced['x_sigma']:.4f}, x_ε = {final_z3_balanced['x_eps']:.4f}")
        print(f"  CFT exact:   x_σ = {x_sigma_exact:.4f}, x_ε = {x_eps_exact:.4f}")

        plain_err_eps = abs(final_plain['x_eps'] - x_eps_exact) / x_eps_exact * 100
        greedy_err_eps = abs(final_z3_greedy['x_eps'] - x_eps_exact) / x_eps_exact * 100
        balanced_err_eps = abs(final_z3_balanced['x_eps'] - x_eps_exact) / x_eps_exact * 100

        print(f"\nErrors (x_ε):")
        print(f"  Plain:       {plain_err_eps:.1f}%")
        print(f"  Z3 Greedy:   {greedy_err_eps:.1f}%")
        print(f"  Z3 Balanced: {balanced_err_eps:.1f}%")

        # Check if fix improves things
        if balanced_err_eps < greedy_err_eps:
            improvement = (greedy_err_eps - balanced_err_eps) / greedy_err_eps * 100
            print(f"\n✓ Balanced allocation improves error by {improvement:.1f}%")
        else:
            print("\n⚠️ Balanced allocation did not improve error")


if __name__ == "__main__":
    main()
