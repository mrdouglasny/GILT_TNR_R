#!/usr/bin/env python3
"""
Debug truncation errors during Z3 vs Plain flow.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np

# Patch gilttnr_step to capture errors
original_import = True

from GiltTNR2D import gilttnr_step as original_gilttnr_step
from GiltTNR2D_Potts import get_initial_tensor_potts_relT, get_scaldims_potts

def run_with_error_tracking(pars, n_steps, name):
    """Run flow and compare Z3 vs plain tensor values."""
    A = get_initial_tensor_potts_relT(pars)
    log_fact = 0.0

    print(f"\n{name}:")
    print(f"{'Step':>4} {'x_ε':>10} {'norm':>15}")
    print("-" * 35)

    for step in range(n_steps + 1):
        sd = get_scaldims_potts(A)
        x_eps = sd[3] if len(sd) > 3 else np.nan

        # Get tensor norm
        if hasattr(A, 'to_ndarray'):
            arr = A.to_ndarray()
            norm = np.linalg.norm(arr)
        else:
            norm = np.nan

        print(f"{step:>4} {x_eps:>10.4f} {norm:>15.6e}")

        if step < n_steps:
            try:
                A, log_fact = original_gilttnr_step(A, log_fact, pars)
            except Exception as e:
                print(f"  Step {step+1} failed: {e}")
                break

    return A


def main():
    print("=" * 70)
    print("Truncation Error Analysis")
    print("=" * 70)

    chi = 16
    n_steps = 5

    base_pars = {
        'q': 3,
        'relT': 1.0,
        'gilt_eps': 1e-6,
        'cg_chis': [chi],
        'cg_eps': 1e-10,
        'verbosity': 0,
    }

    # Plain tensor
    pars_plain = dict(base_pars)
    pars_plain['symmetry_tensors'] = False
    pars_plain['balanced_sectors'] = False
    run_with_error_tracking(pars_plain, n_steps, "Plain Tensor")

    # Z3 greedy
    pars_z3 = dict(base_pars)
    pars_z3['symmetry_tensors'] = True
    pars_z3['balanced_sectors'] = False
    run_with_error_tracking(pars_z3, n_steps, "TensorZ3 (Greedy)")

    # Check the actual tensor entries at step 1
    print("\n" + "=" * 70)
    print("Tensor Comparison at Step 1")
    print("=" * 70)

    A_plain = get_initial_tensor_potts_relT(pars_plain)
    A_z3 = get_initial_tensor_potts_relT(pars_z3)

    A_plain, _ = original_gilttnr_step(A_plain, 0.0, pars_plain)
    A_z3, _ = original_gilttnr_step(A_z3, 0.0, pars_z3)

    arr_plain = A_plain.to_ndarray()
    arr_z3 = A_z3.to_ndarray()

    print(f"\nPlain tensor shape: {arr_plain.shape}")
    print(f"Z3 tensor shape: {arr_z3.shape}")

    # Compare difference
    diff = np.linalg.norm(arr_plain - arr_z3)
    print(f"\nNorm difference: {diff:.6e}")
    print(f"Relative diff: {diff / np.linalg.norm(arr_plain):.6e}")


if __name__ == "__main__":
    main()
