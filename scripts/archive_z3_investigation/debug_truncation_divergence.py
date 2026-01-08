#!/usr/bin/env python3
"""
Debug script to understand exactly where TensorZ3 and plain tensor diverge.

Key question: After step 1, Z3 has shape (10,10,10,10) and plain has (12,10,12,10).
Let's trace through step 2 to see where the divergence in traces comes from.

Hypothesis: The issue is not the trace computation itself, but the fact that
different truncation dimensions lead to different tensor values, which then
produce different environments.
"""

import sys
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr/GiltTNR')
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr')

import numpy as np
from GiltTNR2D import gilttnr_step
from GiltTNR2D_Potts import get_initial_tensor_potts, get_initial_tensor_potts_z3
from tensors.symmetrytensors import TensorZ3
from tensors.tensor import Tensor

def get_initial_tensor(beta, N=3, symmetry_tensors=False):
    """Get initial Potts tensor."""
    pars = {"beta": beta}
    if symmetry_tensors:
        return get_initial_tensor_potts_z3(pars)
    else:
        return get_initial_tensor_potts(pars)

def get_envspec_details(A, Rp, leg, verbosity=0):
    """Get detailed environment spectrum info."""
    from GiltTNR2D_Potts import build_gilt_plaq_op, get_envspec

    plaq = build_gilt_plaq_op(A, A, A, A)
    return get_envspec(plaq, Rp, leg, verbosity=verbosity)

def compare_tensors_detailed(A_z3, A_plain, step_num):
    """Compare two tensors in detail."""
    print(f"\n{'='*60}")
    print(f"Step {step_num} tensor comparison")
    print(f"{'='*60}")

    # Get arrays
    if hasattr(A_z3, 'sects'):
        arr_z3 = A_z3.to_ndarray()
    else:
        arr_z3 = A_z3

    if hasattr(A_plain, 'sects'):
        arr_plain = A_plain.to_ndarray()
    else:
        arr_plain = A_plain

    print(f"Z3 shape: {arr_z3.shape}")
    print(f"Plain shape: {arr_plain.shape}")

    # Compare norms
    norm_z3 = np.linalg.norm(arr_z3)
    norm_plain = np.linalg.norm(arr_plain)
    print(f"Z3 norm: {norm_z3:.10f}")
    print(f"Plain norm: {norm_plain:.10f}")

    # If same shape, compare directly
    if arr_z3.shape == arr_plain.shape:
        diff = np.linalg.norm(arr_z3 - arr_plain)
        rel_diff = diff / norm_plain
        print(f"Direct difference: ||Z3 - Plain|| / ||Plain|| = {rel_diff:.2e}")
        return rel_diff
    else:
        print("Cannot compare directly - different shapes")

        # Compare overlapping region
        min_shape = tuple(min(s1, s2) for s1, s2 in zip(arr_z3.shape, arr_plain.shape))
        print(f"Comparing overlapping region: {min_shape}")

        z3_sub = arr_z3[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]]
        plain_sub = arr_plain[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]]

        diff = np.linalg.norm(z3_sub - plain_sub)
        rel_diff = diff / np.linalg.norm(plain_sub)
        print(f"Overlap region difference: {rel_diff:.2e}")

        # What fraction of norm is in the overlap?
        norm_overlap_z3 = np.linalg.norm(z3_sub)
        norm_overlap_plain = np.linalg.norm(plain_sub)
        print(f"Z3 norm in overlap: {norm_overlap_z3/norm_z3*100:.1f}%")
        print(f"Plain norm in overlap: {norm_overlap_plain/norm_plain*100:.1f}%")

        return None

def run_and_compare_detailed():
    """Run both versions and compare at each step."""

    # Parameters
    beta_c = np.log(1 + np.sqrt(3))
    N = 3
    pars = {
        "gilt_eps": 1e-7,
        "cg_chis": list(range(2, 25)),
        "cg_eps": 1e-10,
        "verbosity": 0,
        "symmetry_tensors": True
    }
    pars_plain = dict(pars)
    pars_plain["symmetry_tensors"] = False

    # Initial tensors
    print("Creating initial tensors...")
    A_z3 = get_initial_tensor(beta_c, N=N, symmetry_tensors=True)
    A_plain = get_initial_tensor(beta_c, N=N, symmetry_tensors=False)

    log_z3 = 0.0
    log_plain = 0.0

    # Compare initial
    compare_tensors_detailed(A_z3, A_plain, 0)

    # Run steps
    for step in range(1, 4):
        print(f"\n{'#'*60}")
        print(f"Running step {step}")
        print(f"{'#'*60}")

        # Run Z3 version
        print(f"\nRunning Z3 GILT-TNR step {step}...")
        try:
            A_z3_new, log_z3_new = gilttnr_step(A_z3, log_z3, pars)
            A_z3, log_z3 = A_z3_new, log_z3_new
            print(f"  Z3 shape after step {step}: {A_z3.shape}")
        except Exception as e:
            print(f"  Z3 step {step} failed: {e}")
            break

        # Run plain version
        print(f"\nRunning plain GILT-TNR step {step}...")
        try:
            A_plain_new, log_plain_new = gilttnr_step(A_plain, log_plain, pars_plain)
            A_plain, log_plain = A_plain_new, log_plain_new
            print(f"  Plain shape after step {step}: {A_plain.shape}")
        except Exception as e:
            print(f"  Plain step {step} failed: {e}")
            break

        # Compare tensors
        compare_tensors_detailed(A_z3, A_plain, step)

        # Compare traces
        print(f"\nComparing traces for step {step}:")

        # For Z3, need to build Rp in Z3 format
        from GiltTNR2D_Potts import build_Rp
        Rp_z3 = build_Rp(A_z3, pars)
        Rp_plain = build_Rp(A_plain, pars_plain)

        # Get envspec for both
        for leg in ['l', 'r', 'u', 'd']:
            try:
                S_z3, U_z3, t_z3, _ = get_envspec_details(A_z3, Rp_z3, leg)
                S_plain, U_plain, t_plain, _ = get_envspec_details(A_plain, Rp_plain, leg)

                # Sort by singular value magnitude
                idx_z3 = np.argsort(-np.abs(S_z3))
                idx_plain = np.argsort(-np.abs(S_plain))

                t_z3_sorted = np.array([t_z3[i] for i in idx_z3])
                t_plain_sorted = np.array([t_plain[i] for i in idx_plain])

                print(f"  Leg {leg}:")
                print(f"    S_z3 len={len(S_z3)}, S_plain len={len(S_plain)}")
                print(f"    Top 5 |t_z3|: {np.abs(t_z3_sorted[:5])}")
                print(f"    Top 5 |t_plain|: {np.abs(t_plain_sorted[:5])}")

                # Check for CDL modes
                cdl_z3 = np.sum(np.abs(t_z3) > 0.1)
                cdl_plain = np.sum(np.abs(t_plain) > 0.1)
                print(f"    CDL modes (|t|>0.1): Z3={cdl_z3}, Plain={cdl_plain}")

            except Exception as e:
                print(f"  Leg {leg}: Error - {e}")

def investigate_trg_step():
    """Look at what happens during the TRG step specifically."""
    print("\n" + "="*60)
    print("Investigating TRG step in isolation")
    print("="*60)

    from GiltTNR2D_Potts import trg_step, get_initial_tensor

    beta_c = np.log(1 + np.sqrt(3))
    N = 3

    # Initial tensors
    A_z3 = get_initial_tensor(beta_c, N=N, symmetry_tensors=True)
    A_plain = get_initial_tensor(beta_c, N=N, symmetry_tensors=False)

    pars_z3 = {"cg_chis": list(range(2, 25)), "cg_eps": 1e-10, "symmetry_tensors": True}
    pars_plain = {"cg_chis": list(range(2, 25)), "cg_eps": 1e-10, "symmetry_tensors": False}

    # TRG step only (no GILT)
    print("\nApplying TRG step (no GILT filtering)...")

    A_z3_trg, log_z3 = trg_step(A_z3, 0.0, pars_z3)
    A_plain_trg, log_plain = trg_step(A_plain, 0.0, pars_plain)

    print(f"After TRG: Z3 shape = {A_z3_trg.shape}, Plain shape = {A_plain_trg.shape}")

    # Compare
    arr_z3 = A_z3_trg.to_ndarray() if hasattr(A_z3_trg, 'sects') else A_z3_trg
    arr_plain = A_plain_trg.to_ndarray() if hasattr(A_plain_trg, 'sects') else A_plain_trg

    if arr_z3.shape == arr_plain.shape:
        diff = np.linalg.norm(arr_z3 - arr_plain) / np.linalg.norm(arr_plain)
        print(f"Relative difference: {diff:.2e}")
    else:
        print("Different shapes - truncation differs!")

        # Look at the SVD singular values
        print("\nInvestigating SVD truncation...")

        # Reshape for SVD (combine legs 0,1 and 2,3)
        chi0 = A_z3.shape[0]
        arr_z3_init = A_z3.to_ndarray() if hasattr(A_z3, 'sects') else A_z3
        arr_plain_init = A_plain.to_ndarray() if hasattr(A_plain, 'sects') else A_plain

        # SVD on the initial tensor
        mat_z3 = arr_z3_init.reshape(chi0*chi0, chi0*chi0)
        mat_plain = arr_plain_init.reshape(chi0*chi0, chi0*chi0)

        _, S_z3_full, _ = np.linalg.svd(mat_z3)
        _, S_plain_full, _ = np.linalg.svd(mat_plain)

        print(f"Initial tensor SVD:")
        print(f"  Z3 top 10 S: {S_z3_full[:10]}")
        print(f"  Plain top 10 S: {S_plain_full[:10]}")
        print(f"  Difference: {np.linalg.norm(S_z3_full[:10] - S_plain_full[:10]):.2e}")

if __name__ == "__main__":
    run_and_compare_detailed()
    investigate_trg_step()
