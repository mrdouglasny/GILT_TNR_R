#!/usr/bin/env python3
"""
Compare GILT after one pure TRG step (no GILT in first step).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D import gilttnr_step, get_envspec, optimize_Rp
from GiltTNR2D_Potts import get_initial_tensor_potts_relT

def main():
    print("=" * 80)
    print("GILT Analysis After One TRG Step (No GILT)")
    print("=" * 80)

    chi = 16
    gilt_eps = 1e-6

    # First run TRG without GILT
    pars_no_gilt = {
        'q': 3, 'relT': 1.0, 'gilt_eps': 0,  # NO GILT
        'cg_chis': [chi], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': False,
    }
    pars_no_gilt_z3 = dict(pars_no_gilt)
    pars_no_gilt_z3['symmetry_tensors'] = True

    # Get initial tensors
    A_plain_0 = get_initial_tensor_potts_relT(pars_no_gilt)
    A_z3_0 = get_initial_tensor_potts_relT(pars_no_gilt_z3)

    # One TRG step (no GILT)
    A_plain_1, log_p = gilttnr_step(A_plain_0, 0.0, pars_no_gilt)
    A_z3_1, log_z = gilttnr_step(A_z3_0, 0.0, pars_no_gilt_z3)

    arr_plain_1 = A_plain_1.to_ndarray()
    arr_z3_1 = A_z3_1.to_ndarray()

    print(f"\nAfter one TRG step (no GILT):")
    print(f"  Plain shape: {arr_plain_1.shape}")
    print(f"  Z3 shape:    {arr_z3_1.shape}")
    print(f"  Plain norm:  {np.linalg.norm(arr_plain_1):.4e}")
    print(f"  Z3 norm:     {np.linalg.norm(arr_z3_1):.4e}")

    # Now analyze the GILT environment at this point
    print("\n" + "-" * 80)
    print("GILT Environment Analysis (if we were to apply GILT now)")
    print("-" * 80)

    # Use gilt_eps for analysis
    pars_gilt = dict(pars_no_gilt)
    pars_gilt['gilt_eps'] = gilt_eps
    pars_gilt_z3 = dict(pars_no_gilt_z3)
    pars_gilt_z3['gilt_eps'] = gilt_eps

    U_plain, S_plain = get_envspec(A_plain_1, A_plain_1, pars_gilt, where="S")
    U_z3, S_z3 = get_envspec(A_z3_1, A_z3_1, pars_gilt_z3, where="S")

    arr_U_plain = U_plain.to_ndarray()
    arr_U_z3 = U_z3.to_ndarray()

    S_plain_arr = S_plain.to_ndarray()
    S_z3_arr = S_z3.to_ndarray()

    print(f"\nU shapes: plain={arr_U_plain.shape}, Z3={arr_U_z3.shape}")
    print(f"S lengths: plain={len(S_plain_arr)}, Z3={len(S_z3_arr)}")

    # Traces
    t_plain = ncon(U_plain, [1, 1, -1]).to_ndarray()
    t_z3 = ncon(U_z3, [1, 1, -1]).to_ndarray()

    chi_plain = arr_U_plain.shape[0]
    chi_z3 = arr_U_z3.shape[0]
    max_trace_plain = np.sqrt(chi_plain)
    max_trace_z3 = np.sqrt(chi_z3)

    print(f"\nMax possible trace: plain={max_trace_plain:.2f}, Z3={max_trace_z3:.2f}")

    # Mode analysis
    print(f"\n{'Mode':<6} {'S_plain':<14} {'S_z3':<14} {'|t|_plain':<12} {'|t|_z3':<12} {'|t|/√χ plain':<14} {'|t|/√χ Z3':<14}")
    print("-" * 100)

    max_modes = min(15, len(t_plain), len(t_z3))
    for i in range(max_modes):
        sp = S_plain_arr[i] if i < len(S_plain_arr) else 0
        sz = S_z3_arr[i] if i < len(S_z3_arr) else 0
        tp = abs(t_plain[i]) if i < len(t_plain) else 0
        tz = abs(t_z3[i]) if i < len(t_z3) else 0
        rel_p = tp / max_trace_plain
        rel_z = tz / max_trace_z3
        print(f"{i:<6} {sp:<14.2e} {sz:<14.2e} {tp:<12.4f} {tz:<12.4f} {rel_p:<14.4f} {rel_z:<14.4f}")

    # Get Rp filters
    Rp_plain, err_plain = optimize_Rp(U_plain, S_plain, pars_gilt)
    Rp_z3, err_z3 = optimize_Rp(U_z3, S_z3, pars_gilt_z3)

    arr_Rp_plain = Rp_plain.to_ndarray()
    arr_Rp_z3 = Rp_z3.to_ndarray()

    print(f"\n" + "-" * 80)
    print(f"Rp Filter Analysis")
    print("-" * 80)

    print(f"\nRp shapes: plain={arr_Rp_plain.shape}, Z3={arr_Rp_z3.shape}")

    # SVD of Rp
    S_Rp_plain = np.linalg.svd(arr_Rp_plain, compute_uv=False)
    S_Rp_z3 = np.linalg.svd(arr_Rp_z3, compute_uv=False)

    print(f"\nRp singular values:")
    print(f"  Plain: {S_Rp_plain}")
    print(f"  Z3:    {S_Rp_z3}")

    # Distance from identity
    I_plain = np.eye(arr_Rp_plain.shape[0])
    I_z3 = np.eye(arr_Rp_z3.shape[0])

    dist_plain = np.linalg.norm(arr_Rp_plain - I_plain)
    dist_z3 = np.linalg.norm(arr_Rp_z3 - I_z3)

    print(f"\n||Rp - I|| (GILT filtering strength):")
    print(f"  Plain: {dist_plain:.4f}")
    print(f"  Z3:    {dist_z3:.4f}")

    if abs(dist_plain - dist_z3) > 0.1:
        print(f"\n⚠️  Significant difference in GILT filtering!")
        print(f"   Ratio Z3/Plain: {dist_z3/dist_plain:.2f}")

    # Analyze which modes contribute
    print(f"\n" + "-" * 80)
    print("Mode Contributions to GILT Filter")
    print("-" * 80)

    ratio_plain = S_plain_arr / gilt_eps
    weight_plain = ratio_plain**2 / (1 + ratio_plain**2)

    ratio_z3 = S_z3_arr / gilt_eps
    weight_z3 = ratio_z3**2 / (1 + ratio_z3**2)

    # Weighted trace contributions
    contrib_plain = np.abs(t_plain) * weight_plain[:len(t_plain)]
    contrib_z3 = np.abs(t_z3) * weight_z3[:len(t_z3)]

    print(f"\nModes with significant contribution (>0.1):")
    print(f"\nPlain:")
    for i, c in enumerate(contrib_plain):
        if c > 0.1:
            print(f"  Mode {i}: contrib={c:.4f}, |t|={abs(t_plain[i]):.4f}, S={S_plain_arr[i]:.2e}")

    print(f"\nZ3:")
    for i, c in enumerate(contrib_z3):
        if c > 0.1:
            print(f"  Mode {i}: contrib={c:.4f}, |t|={abs(t_z3[i]):.4f}, S={S_z3_arr[i]:.2e}")

    total_plain = sum(contrib_plain)
    total_z3 = sum(contrib_z3)
    print(f"\nTotal contribution: plain={total_plain:.4f}, Z3={total_z3:.4f}")


if __name__ == "__main__":
    main()
