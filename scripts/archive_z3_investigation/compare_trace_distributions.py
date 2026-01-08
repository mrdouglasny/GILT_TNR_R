#!/usr/bin/env python3
"""
Compare trace distributions between plain and Z3 in detail.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D import gilttnr_step, get_envspec
from GiltTNR2D_Potts import get_initial_tensor_potts_relT


def main():
    print("=" * 70)
    print("Trace Distribution Comparison")
    print("=" * 70)

    chi = 16
    gilt_eps = 1e-6

    pars_plain = {
        'q': 3, 'relT': 1.0, 'gilt_eps': gilt_eps,
        'cg_chis': [chi], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': False,
    }
    pars_z3 = dict(pars_plain)
    pars_z3['symmetry_tensors'] = True

    # Get tensors at step 2
    A_plain = get_initial_tensor_potts_relT(pars_plain)
    A_z3 = get_initial_tensor_potts_relT(pars_z3)

    for step in range(2):
        A_plain, _ = gilttnr_step(A_plain, 0.0, pars_plain)
        A_z3, _ = gilttnr_step(A_z3, 0.0, pars_z3)

    # Get environment
    U_plain, S_plain = get_envspec(A_plain, A_plain, pars_plain, where="S")
    U_z3, S_z3 = get_envspec(A_z3, A_z3, pars_z3, where="S")

    t_plain = ncon(U_plain, [1, 1, -1]).to_ndarray()
    t_z3 = ncon(U_z3, [1, 1, -1]).to_ndarray()

    S_plain_arr = S_plain.to_ndarray()
    S_z3_arr = S_z3.to_ndarray()

    print(f"\nTotal modes: plain={len(t_plain)}, Z3={len(t_z3)}")

    # Histogram of |t| values
    print("\n--- Trace Magnitude Distribution ---")
    bins = [0, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    print(f"{'Bin':<15} {'Plain':<10} {'Z3':<10}")
    print("-" * 35)
    for i in range(len(bins) - 1):
        n_plain = np.sum((np.abs(t_plain) >= bins[i]) & (np.abs(t_plain) < bins[i+1]))
        n_z3 = np.sum((np.abs(t_z3) >= bins[i]) & (np.abs(t_z3) < bins[i+1]))
        print(f"[{bins[i]:.2f}, {bins[i+1]:.2f})" + f"     {n_plain:<10} {n_z3:<10}")

    # The key question: Are the singular values similar but traces different?
    print("\n--- Singular Value Comparison ---")
    print("Sorted singular values (normalized by max):")
    S_plain_norm = S_plain_arr / S_plain_arr[0]
    S_z3_norm = S_z3_arr / S_z3_arr[0]

    print(f"{'Rank':<6} {'S_plain':<12} {'S_z3':<12} {'|t|_plain':<12} {'|t|_z3':<12}")
    print("-" * 54)
    for i in range(min(20, len(S_plain_arr), len(S_z3_arr))):
        sp = S_plain_norm[i] if i < len(S_plain_norm) else 0
        sz = S_z3_norm[i] if i < len(S_z3_norm) else 0
        tp = np.abs(t_plain[i]) if i < len(t_plain) else 0
        tz = np.abs(t_z3[i]) if i < len(t_z3) else 0
        print(f"{i:<6} {sp:<12.4f} {sz:<12.4f} {tp:<12.4f} {tz:<12.4f}")

    # Sum of |t| weighted by S (this is what GILT uses)
    print("\n--- Weighted Trace Analysis ---")
    # GILT uses weight = (S/eps)^2 / (1 + (S/eps)^2)
    ratio_plain = S_plain_arr / gilt_eps
    weight_plain = ratio_plain**2 / (1 + ratio_plain**2)

    ratio_z3 = S_z3_arr / gilt_eps
    weight_z3 = ratio_z3**2 / (1 + ratio_z3**2)

    # Weighted contributions
    contrib_plain = np.abs(t_plain) * weight_plain[:len(t_plain)]
    contrib_z3 = np.abs(t_z3) * weight_z3[:len(t_z3)]

    print(f"Sum of |t| * weight:")
    print(f"  Plain: {np.sum(contrib_plain):.4f}")
    print(f"  Z3:    {np.sum(contrib_z3):.4f}")

    # Number of significant contributions
    print(f"\nModes with |t| * weight > 0.1:")
    print(f"  Plain: {np.sum(contrib_plain > 0.1)}")
    print(f"  Z3:    {np.sum(contrib_z3 > 0.1)}")

    print("\n--- Top Contributing Modes ---")
    idx_plain = np.argsort(contrib_plain)[::-1]
    idx_z3 = np.argsort(contrib_z3)[::-1]

    print("\nPlain top 10:")
    for i in range(10):
        idx = idx_plain[i]
        print(f"  Mode {idx}: S={S_plain_arr[idx]:.2e}, |t|={np.abs(t_plain[idx]):.4f}, weight={weight_plain[idx]:.4f}, contrib={contrib_plain[idx]:.4f}")

    print("\nZ3 top 10:")
    for i in range(10):
        idx = idx_z3[i]
        print(f"  Mode {idx}: S={S_z3_arr[idx]:.2e}, |t|={np.abs(t_z3[idx]):.4f}, weight={weight_z3[idx]:.4f}, contrib={contrib_z3[idx]:.4f}")


if __name__ == "__main__":
    main()
