#!/usr/bin/env python3
"""
Debug GILT after one TRG step - where truncation starts mattering.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D import gilttnr_step, get_envspec, optimize_Rp
from GiltTNR2D_Potts import get_initial_tensor_potts_relT

def main():
    print("=" * 70)
    print("GILT Analysis After One TRG Step")
    print("=" * 70)

    chi = 16

    pars_plain = {
        'q': 3,
        'relT': 1.0,
        'gilt_eps': 1e-6,
        'cg_chis': [chi],
        'cg_eps': 1e-10,
        'verbosity': 0,
        'symmetry_tensors': False,
    }
    pars_z3 = dict(pars_plain)
    pars_z3['symmetry_tensors'] = True

    # Initial tensors
    A_plain_0 = get_initial_tensor_potts_relT(pars_plain)
    A_z3_0 = get_initial_tensor_potts_relT(pars_z3)

    # One TRG step
    A_plain_1, _ = gilttnr_step(A_plain_0, 0.0, pars_plain)
    A_z3_1, _ = gilttnr_step(A_z3_0, 0.0, pars_z3)

    print(f"\nAfter step 1:")
    print(f"  Plain shape: {A_plain_1.to_ndarray().shape}")
    print(f"  Z3 shape: {A_z3_1.shape}")

    # Now analyze GILT environment at step 1
    print("\n--- GILT Environment After Step 1 ---")

    U_plain, S_plain = get_envspec(A_plain_1, A_plain_1, pars_plain, where="S")
    U_z3, S_z3 = get_envspec(A_z3_1, A_z3_1, pars_z3, where="S")

    print(f"\nPlain U shape: {U_plain.shape}")
    print(f"Z3 U shape: {U_z3.shape}")

    # Singular values
    S_plain_arr = S_plain.to_ndarray()
    S_z3_arr = S_z3.to_ndarray()

    print(f"\nPlain S (top 10): {S_plain_arr[:10] / S_plain_arr[0]}")
    print(f"Z3 S (top 10):    {S_z3_arr[:10] / S_z3_arr[0]}")

    # Traces
    t_plain = ncon(U_plain, [1, 1, -1]).to_ndarray()
    t_z3 = ncon(U_z3, [1, 1, -1]).to_ndarray()

    # Normalize traces by Frobenius norm of each U slice
    arr_U_plain = U_plain.to_ndarray()
    arr_U_z3 = U_z3.to_ndarray()

    chi_plain = arr_U_plain.shape[0]
    chi_z3 = arr_U_z3.shape[0]
    max_trace_plain = np.sqrt(chi_plain)
    max_trace_z3 = np.sqrt(chi_z3)

    print(f"\nMax possible trace: plain={max_trace_plain:.2f}, z3={max_trace_z3:.2f}")

    # Identity-likeness of top modes
    print(f"\nIdentity-likeness (|trace|/sqrt(chi)):")
    print("Plain:", end=" ")
    for i in range(min(5, len(t_plain))):
        il = abs(t_plain[i]) / max_trace_plain
        print(f"{il:.3f}", end=" ")
    print()

    print("Z3:   ", end=" ")
    for i in range(min(5, len(t_z3))):
        il = abs(t_z3[i]) / max_trace_z3
        print(f"{il:.3f}", end=" ")
    print()

    # Non-zero traces
    thresh = 0.1 * max_trace_plain
    n_nonzero_plain = np.sum(np.abs(t_plain) > thresh)
    n_nonzero_z3 = np.sum(np.abs(t_z3) > thresh)

    print(f"\nNumber of identity-like modes (|t| > 0.1*sqrt(chi)):")
    print(f"  Plain: {n_nonzero_plain}")
    print(f"  Z3:    {n_nonzero_z3}")

    # Build and compare Rp filters
    print("\n--- Rp Filters ---")

    Rp_plain, err_plain = optimize_Rp(U_plain, S_plain, pars_plain)
    Rp_z3, err_z3 = optimize_Rp(U_z3, S_z3, pars_z3)

    arr_Rp_plain = Rp_plain.to_ndarray()
    arr_Rp_z3 = Rp_z3.to_ndarray()

    print(f"Plain Rp shape: {arr_Rp_plain.shape}")
    print(f"Z3 Rp shape:    {arr_Rp_z3.shape}")

    # SVD of Rp - tells us how GILT modifies the bond
    S_Rp_plain = np.linalg.svd(arr_Rp_plain, compute_uv=False)
    S_Rp_z3 = np.linalg.svd(arr_Rp_z3, compute_uv=False)

    print(f"\nRp singular values (how GILT modifies bond):")
    print(f"Plain: {S_Rp_plain}")
    print(f"Z3:    {S_Rp_z3}")

    # Distance from identity
    I_plain = np.eye(arr_Rp_plain.shape[0])
    I_z3 = np.eye(arr_Rp_z3.shape[0])

    dist_plain = np.linalg.norm(arr_Rp_plain - I_plain)
    dist_z3 = np.linalg.norm(arr_Rp_z3 - I_z3)

    print(f"\n||Rp - I||:")
    print(f"  Plain: {dist_plain:.6f}")
    print(f"  Z3:    {dist_z3:.6f}")

    # Check the actual filtering effect
    print("\n--- Filtering Analysis ---")

    gilt_eps = pars_plain['gilt_eps']

    # For plain
    ratio_plain = S_plain_arr / gilt_eps
    weight_plain = ratio_plain**2 / (1 + ratio_plain**2)
    tp_plain = t_plain * weight_plain

    # Contribution of each mode to Rp
    contrib_plain = np.abs(tp_plain)

    print(f"\nPlain mode contributions to Rp filter:")
    for i in range(min(10, len(contrib_plain))):
        if contrib_plain[i] > 1e-3:
            print(f"  Mode {i}: |tp|={contrib_plain[i]:.4f}, S={S_plain_arr[i]:.2e}, weight={weight_plain[i]:.4f}")

    # For Z3
    ratio_z3 = S_z3_arr / gilt_eps
    weight_z3 = ratio_z3**2 / (1 + ratio_z3**2)
    tp_z3 = t_z3 * weight_z3

    contrib_z3 = np.abs(tp_z3)

    print(f"\nZ3 mode contributions to Rp filter:")
    for i in range(min(10, len(contrib_z3))):
        if contrib_z3[i] > 1e-3:
            print(f"  Mode {i}: |tp|={contrib_z3[i]:.4f}, S={S_z3_arr[i]:.2e}, weight={weight_z3[i]:.4f}")

if __name__ == "__main__":
    main()
