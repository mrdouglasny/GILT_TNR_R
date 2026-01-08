#!/usr/bin/env python3
"""
Debug GILT step for Z3 vs plain tensors.

Compare what happens in the GILT algorithm:
1. Environment computation
2. Trace computation
3. Filter matrix R'
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D import gilttnr_step, gilt_plaq
from GiltTNR2D_Potts import get_initial_tensor_potts_relT, get_scaldims_potts

def analyze_after_gilt(A_plain, A_z3, pars_plain, pars_z3):
    """Compare tensors after GILT filtering only (no TRG)."""

    print("\n--- GILT plaq analysis ---")

    # Apply gilt_plaq to each
    pars_plain_gilt = dict(pars_plain)
    pars_plain_gilt['verbosity'] = 2  # Get detailed output

    pars_z3_gilt = dict(pars_z3)
    pars_z3_gilt['verbosity'] = 2

    print("\nPlain tensor GILT:")
    try:
        # gilt_plaq expects two tensors (A1, A2 = A, A for square lattice)
        A1_plain_gilt, A2_plain_gilt = gilt_plaq(A_plain, A_plain, pars_plain_gilt)
        print(f"  After GILT: shape = {A1_plain_gilt.to_ndarray().shape}")
    except Exception as e:
        print(f"  GILT failed: {e}")
        A1_plain_gilt = None

    print("\nZ3 tensor GILT:")
    try:
        A1_z3_gilt, A2_z3_gilt = gilt_plaq(A_z3, A_z3, pars_z3_gilt)
        print(f"  After GILT: shape = {A1_z3_gilt.shape}")
    except Exception as e:
        print(f"  GILT failed: {e}")
        import traceback
        traceback.print_exc()
        A1_z3_gilt = None

    return A1_plain_gilt, A1_z3_gilt


def main():
    print("=" * 70)
    print("GILT Step Debugging: Z3 vs Plain")
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

    print(f"\nParameters: χ={chi}, gilt_eps=1e-6")

    # Get initial tensors
    A_plain = get_initial_tensor_potts_relT(pars_plain)
    A_z3 = get_initial_tensor_potts_relT(pars_z3)

    print(f"\nInitial plain tensor: shape = {A_plain.to_ndarray().shape}")
    print(f"Initial Z3 tensor: shape = {A_z3.shape}")

    # Check scaling dimensions before any RG
    sd_plain = get_scaldims_potts(A_plain)
    sd_z3 = get_scaldims_potts(A_z3)
    print(f"\nInitial scaling dims (plain): {sd_plain[:5]}")
    print(f"Initial scaling dims (Z3):    {sd_z3[:5]}")

    # Do one gilttnr step
    print("\n" + "=" * 70)
    print("After ONE gilttnr_step")
    print("=" * 70)

    A_plain_1, log_plain = gilttnr_step(A_plain, 0.0, pars_plain)
    A_z3_1, log_z3 = gilttnr_step(A_z3, 0.0, pars_z3)

    print(f"\nPlain tensor after step 1: shape = {A_plain_1.to_ndarray().shape}")
    print(f"Z3 tensor after step 1: shape = {A_z3_1.shape}")

    sd_plain_1 = get_scaldims_potts(A_plain_1)
    sd_z3_1 = get_scaldims_potts(A_z3_1)
    print(f"\nScaling dims (plain): {sd_plain_1[:5]}")
    print(f"Scaling dims (Z3):    {sd_z3_1[:5]}")

    # Compare tensor elements
    print("\n--- Tensor element comparison ---")
    arr_plain = A_plain_1.to_ndarray()
    arr_z3 = A_z3_1.to_ndarray()

    print(f"Plain shape: {arr_plain.shape}")
    print(f"Z3 shape:    {arr_z3.shape}")

    if arr_plain.shape == arr_z3.shape:
        diff = np.linalg.norm(arr_plain - arr_z3)
        print(f"||plain - z3||: {diff:.6e}")
        print(f"Relative diff: {diff / np.linalg.norm(arr_plain):.6e}")
    else:
        print("Shapes differ, can't directly compare")

    # SVD spectrum comparison
    print("\n--- SVD spectrum after step 1 ---")
    U_p, S_p, V_p = np.linalg.svd(arr_plain.reshape(-1, arr_plain.shape[-1]))
    S_p = S_p / S_p[0]

    if arr_z3.shape == arr_plain.shape:
        U_z, S_z, V_z = np.linalg.svd(arr_z3.reshape(-1, arr_z3.shape[-1]))
        S_z = S_z / S_z[0]
        print(f"Plain: {S_p[:8]}")
        print(f"Z3:    {S_z[:8]}")
    else:
        # Reshape Z3 tensor
        n = arr_z3.shape[0] * arr_z3.shape[1]
        m = arr_z3.shape[2] * arr_z3.shape[3]
        U_z, S_z, V_z = np.linalg.svd(arr_z3.reshape(n, m))
        S_z = S_z / S_z[0]
        print(f"Plain: {S_p[:8]}")
        print(f"Z3:    {S_z[:8]}")

    # Step 2 - where divergence typically starts
    print("\n" + "=" * 70)
    print("After TWO gilttnr_steps")
    print("=" * 70)

    A_plain_2, log_plain = gilttnr_step(A_plain_1, log_plain, pars_plain)
    A_z3_2, log_z3 = gilttnr_step(A_z3_1, log_z3, pars_z3)

    sd_plain_2 = get_scaldims_potts(A_plain_2)
    sd_z3_2 = get_scaldims_potts(A_z3_2)
    print(f"Scaling dims (plain): {sd_plain_2[:5]}")
    print(f"Scaling dims (Z3):    {sd_z3_2[:5]}")

    print(f"\nPlain x_ε = {sd_plain_2[3]:.4f} (CFT: 0.8)")
    print(f"Z3 x_ε = {sd_z3_2[3]:.4f} (CFT: 0.8)")

    # Check sector dimensions of Z3 tensor
    print(f"\nZ3 tensor shape after step 2: {A_z3_2.shape}")
    print(f"Z3 tensor qhape after step 2: {A_z3_2.qhape}")


if __name__ == "__main__":
    main()
