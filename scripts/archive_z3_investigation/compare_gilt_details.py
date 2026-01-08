#!/usr/bin/env python3
"""
Detailed comparison of GILT filtering step by step.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D import get_envspec, optimize_Rp
from GiltTNR2D_Potts import get_initial_tensor_potts_relT

def z3_to_plain(arr_z3):
    """Transform Z3 charge basis tensor to plain spin basis."""
    omega = np.exp(2j * np.pi / 3)
    F = np.array([
        [1, 1, 1],
        [1, omega, omega**2],
        [1, omega**2, omega]
    ]) / np.sqrt(3)
    Fdag = F.T.conj()

    shape = arr_z3.shape
    result = arr_z3.copy()

    # Transform each leg
    result = np.tensordot(Fdag, result, axes=([1], [0]))
    result = np.tensordot(Fdag, result, axes=([1], [1]))
    result = np.moveaxis(result, 0, 1)
    result = np.tensordot(F, result, axes=([1], [2]))
    result = np.moveaxis(result, 0, 2)
    result = np.tensordot(F, result, axes=([1], [3]))
    result = np.moveaxis(result, 0, 3)

    return result


def apply_Rp_to_tensor(A, Rp, leg):
    """Apply Rp filter to a specific leg of tensor A."""
    # Rp is applied as: A' = Rp @ A (on the specified leg)
    # Using ncon for clarity
    if leg == 0:  # West leg
        return ncon([Rp, A], [[-1, 1], [1, -2, -3, -4]])
    elif leg == 1:  # South leg
        return ncon([Rp, A], [[-2, 1], [-1, 1, -3, -4]])
    elif leg == 2:  # East leg
        return ncon([Rp, A], [[-3, 1], [-1, -2, 1, -4]])
    elif leg == 3:  # North leg
        return ncon([Rp, A], [[-4, 1], [-1, -2, -3, 1]])


def main():
    print("=" * 80)
    print("Detailed GILT Filter Comparison")
    print("=" * 80)

    chi = 16
    gilt_eps = 1e-6

    pars_plain = {
        'q': 3, 'relT': 1.0, 'gilt_eps': gilt_eps,
        'cg_chis': [chi], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': False,
    }
    pars_z3 = dict(pars_plain)
    pars_z3['symmetry_tensors'] = True

    A_plain = get_initial_tensor_potts_relT(pars_plain)
    A_z3 = get_initial_tensor_potts_relT(pars_z3)

    arr_plain = A_plain.to_ndarray()
    arr_z3 = A_z3.to_ndarray()

    # Verify tensors are equivalent
    arr_z3_to_plain = z3_to_plain(arr_z3)
    print(f"\nInitial: ||plain - Z3_to_plain||/||plain|| = {np.linalg.norm(arr_plain - arr_z3_to_plain)/np.linalg.norm(arr_plain):.2e}")

    # Get GILT filters
    print("\n" + "-" * 80)
    print("GILT Environment Analysis (South edge)")
    print("-" * 80)

    U_plain, S_plain = get_envspec(A_plain, A_plain, pars_plain, where="S")
    U_z3, S_z3 = get_envspec(A_z3, A_z3, pars_z3, where="S")

    # Traces
    t_plain = ncon(U_plain, [1, 1, -1]).to_ndarray()
    t_z3 = ncon(U_z3, [1, 1, -1]).to_ndarray()

    print(f"\nTraces (identity-likeness):")
    print(f"  Plain: {np.abs(t_plain)}")
    print(f"  Z3:    {np.abs(t_z3)}")

    # Get Rp filters
    Rp_plain, err_plain = optimize_Rp(U_plain, S_plain, pars_plain)
    Rp_z3, err_z3 = optimize_Rp(U_z3, S_z3, pars_z3)

    arr_Rp_plain = Rp_plain.to_ndarray()
    arr_Rp_z3 = Rp_z3.to_ndarray()

    print(f"\nRp filter shapes: plain={arr_Rp_plain.shape}, Z3={arr_Rp_z3.shape}")

    # Check if Rp_z3 is close to identity when transformed
    # For the initial tensor, the filter should be ~ identity at criticality
    I_plain = np.eye(arr_Rp_plain.shape[0])
    I_z3 = np.eye(arr_Rp_z3.shape[0])

    print(f"\n||Rp - I||:")
    print(f"  Plain: {np.linalg.norm(arr_Rp_plain - I_plain):.6f}")
    print(f"  Z3:    {np.linalg.norm(arr_Rp_z3 - I_z3):.6f}")

    # Now check what the Rp SHOULD be based on the formula
    # Rp = I - sum_i w_i * (t_i/|t_i|) * U_i
    print("\n" + "-" * 80)
    print("Mode-by-mode Analysis")
    print("-" * 80)

    S_plain_arr = S_plain.to_ndarray()
    S_z3_arr = S_z3.to_ndarray()

    # Compute weights
    ratio_plain = S_plain_arr / gilt_eps
    weight_plain = ratio_plain**2 / (1 + ratio_plain**2)

    ratio_z3 = S_z3_arr / gilt_eps
    weight_z3 = ratio_z3**2 / (1 + ratio_z3**2)

    print(f"\nMode contributions to Rp filter:")
    print(f"{'Mode':<6} {'|t|_plain':<12} {'|t|_Z3':<12} {'w_plain':<10} {'w_Z3':<10} {'Contrib_plain':<12} {'Contrib_Z3':<12}")
    print("-" * 80)

    for i in range(min(len(t_plain), len(t_z3))):
        tp = abs(t_plain[i])
        tz = abs(t_z3[i])
        wp = weight_plain[i]
        wz = weight_z3[i]
        # Contribution = |t * w| (since t/|t| has magnitude 1)
        cp = tp * wp
        cz = tz * wz
        print(f"{i:<6} {tp:<12.4f} {tz:<12.4f} {wp:<10.4f} {wz:<10.4f} {cp:<12.4f} {cz:<12.4f}")

    # Key question: what modes differ significantly?
    print("\n" + "-" * 80)
    print("Modes with Large Difference in Contribution")
    print("-" * 80)

    for i in range(min(len(t_plain), len(t_z3))):
        tp = abs(t_plain[i])
        tz = abs(t_z3[i])
        wp = weight_plain[i]
        wz = weight_z3[i]
        cp = tp * wp
        cz = tz * wz

        diff = abs(cp - cz)
        if diff > 0.1:
            print(f"Mode {i}: |contrib_plain - contrib_Z3| = {diff:.4f}")
            print(f"  Plain: |t|={tp:.4f}, w={wp:.4f}, contrib={cp:.4f}")
            print(f"  Z3:    |t|={tz:.4f}, w={wz:.4f}, contrib={cz:.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    # Total "filtering strength"
    total_plain = sum(abs(t_plain[i]) * weight_plain[i] for i in range(len(t_plain)))
    total_z3 = sum(abs(t_z3[i]) * weight_z3[i] for i in range(len(t_z3)))

    print(f"\nTotal GILT filtering strength:")
    print(f"  Plain: {total_plain:.4f}")
    print(f"  Z3:    {total_z3:.4f}")
    print(f"  Ratio Z3/Plain: {total_z3/total_plain:.4f}")

    if total_z3 > total_plain:
        print(f"\n⚠️  Z3 GILT is {total_z3/total_plain:.1%} more aggressive!")


if __name__ == "__main__":
    main()
