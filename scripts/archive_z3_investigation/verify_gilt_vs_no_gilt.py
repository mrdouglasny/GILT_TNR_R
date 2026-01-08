#!/usr/bin/env python3
"""
Compare Z3 behavior WITH and WITHOUT GILT filtering.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D import gilttnr_step
from GiltTNR2D_Potts import get_initial_tensor_potts_relT

def compute_scaling_dim(A, log_fact):
    """Compute x_ε from transfer matrix (plain tensor only)."""
    try:
        # Build transfer matrix - only works for plain tensors
        T = ncon([A, A], [[-1, 1, -3, 2], [-2, 1, -4, 2]])
        arr_T = T.to_ndarray()
        chi1, chi2, chi3, chi4 = arr_T.shape
        T_mat = arr_T.reshape(chi1 * chi2, chi3 * chi4)

        # Eigenvalues
        eigs = np.linalg.eigvals(T_mat)
        eigs_sorted = sorted(np.abs(eigs), reverse=True)

        if len(eigs_sorted) >= 2 and eigs_sorted[0] > 1e-10:
            ratio = eigs_sorted[1] / eigs_sorted[0]
            if ratio > 1e-10:
                x_eps = -np.log(ratio) / np.log(2)
                return x_eps
    except (ValueError, TypeError):
        # Z3 tensors may fail this contraction
        pass
    return None


def run_flow(use_gilt, use_z3, n_steps=6):
    """Run RG flow and extract scaling dimension at each step."""
    chi = 16
    gilt_eps = 1e-6 if use_gilt else 0.0

    pars = {
        'q': 3, 'relT': 1.0, 'gilt_eps': gilt_eps,
        'cg_chis': [chi], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': use_z3,
    }

    A = get_initial_tensor_potts_relT(pars)
    log_fact = 0.0

    results = []
    for step in range(1, n_steps + 1):
        A, log_fact = gilttnr_step(A, log_fact, pars)
        x_eps = compute_scaling_dim(A, log_fact)
        arr = A.to_ndarray()
        results.append({
            'step': step,
            'x_eps': x_eps,
            'norm': np.linalg.norm(arr),
            'shape': arr.shape,
        })

    return results


def main():
    print("=" * 80)
    print("Comparing GILT vs No-GILT for Plain and Z3 Tensors")
    print("=" * 80)
    print(f"\nTarget: x_ε = 0.8 (4/5 for 3-state Potts)")

    configs = [
        ("Plain + GILT", True, False),
        ("Plain + No GILT", False, False),
        ("Z3 + GILT", True, True),
        ("Z3 + No GILT", False, True),
    ]

    all_results = {}
    for name, use_gilt, use_z3 in configs:
        print(f"\nRunning: {name}...")
        results = run_flow(use_gilt, use_z3)
        all_results[name] = results

    # Print comparison table
    print("\n" + "=" * 80)
    print("x_ε at Each Step (CFT target = 0.8)")
    print("=" * 80)
    print(f"{'Step':<6}", end="")
    for name, _, _ in configs:
        print(f"{name:<20}", end="")
    print()
    print("-" * 86)

    for step in range(1, 7):
        print(f"{step:<6}", end="")
        for name, _, _ in configs:
            r = all_results[name][step-1]
            if r['x_eps'] is not None:
                err = abs(r['x_eps'] - 0.8) / 0.8 * 100
                print(f"{r['x_eps']:.4f} ({err:>5.1f}%)   ", end="")
            else:
                print(f"{'N/A':<20}", end="")
        print()

    print("\n" + "=" * 80)
    print("Tensor Norms at Each Step")
    print("=" * 80)
    print(f"{'Step':<6}", end="")
    for name, _, _ in configs:
        print(f"{name:<20}", end="")
    print()
    print("-" * 86)

    for step in range(1, 7):
        print(f"{step:<6}", end="")
        for name, _, _ in configs:
            r = all_results[name][step-1]
            print(f"{r['norm']:.4e}          ", end="")
        print()

    print("\n" + "=" * 80)
    print("Analysis")
    print("=" * 80)

    # Check if Z3 + No GILT is stable
    z3_gilt = all_results["Z3 + GILT"]
    z3_no_gilt = all_results["Z3 + No GILT"]
    plain_gilt = all_results["Plain + GILT"]

    print("\nKey observations:")

    # Compare x_eps at step 3 (early, before much drift)
    if z3_gilt[2]['x_eps'] and z3_no_gilt[2]['x_eps']:
        print(f"  Step 3 x_ε: Z3+GILT={z3_gilt[2]['x_eps']:.4f}, Z3+NoGILT={z3_no_gilt[2]['x_eps']:.4f}")

    if z3_gilt[5]['x_eps'] and z3_no_gilt[5]['x_eps']:
        err_gilt = abs(z3_gilt[5]['x_eps'] - 0.8)
        err_no_gilt = abs(z3_no_gilt[5]['x_eps'] - 0.8)
        print(f"  Step 6 x_ε error: Z3+GILT={err_gilt:.2f}, Z3+NoGILT={err_no_gilt:.2f}")
        if err_no_gilt < err_gilt:
            print("  → Z3 is MORE ACCURATE without GILT!")
            print("  → This confirms GILT is causing the Z3 divergence.")
        else:
            print("  → Z3 diverges with or without GILT")
            print("  → Root cause may be in coarse-graining, not GILT")


if __name__ == "__main__":
    main()
