#!/usr/bin/env python3
"""
Compare tensors via Procrustes analysis.

Find min ||T1 - U T2 U†|| over unitary U.
This tells us if two tensors are equivalent up to basis change.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from scipy.linalg import orthogonal_procrustes
from ncon import ncon
from GiltTNR2D import gilttnr_step, get_envspec, optimize_Rp
from GiltTNR2D_Potts import get_initial_tensor_potts_relT


def procrustes_distance(T1, T2):
    """
    Compute min ||T1 - U @ T2 @ U.T|| over orthogonal U.

    For 4-leg tensors, we reshape to matrix and find the best
    orthogonal transformation.
    """
    # Reshape to matrix for Procrustes
    shape1 = T1.shape
    shape2 = T2.shape

    if shape1 != shape2:
        return float('inf'), None

    # Reshape to matrix (combine first two legs and last two legs)
    d1 = shape1[0] * shape1[1]
    d2 = shape1[2] * shape1[3]

    M1 = T1.reshape(d1, d2)
    M2 = T2.reshape(d1, d2)

    # Orthogonal Procrustes: find U that minimizes ||M1 - U @ M2||
    # scipy.linalg.orthogonal_procrustes finds R such that ||A - B @ R|| is minimized
    # We want ||M1 - U @ M2||, so we compute ||M1.T - M2.T @ U.T||
    # which gives us U.T, then we transpose

    try:
        # Method: SVD of M1 @ M2.T
        # U_opt = U @ V.T where M1 @ M2.T = U @ S @ V.T
        U, s, Vh = np.linalg.svd(M1 @ M2.T.conj())
        U_opt = U @ Vh

        # Compute distance
        diff = M1 - U_opt @ M2
        dist = np.linalg.norm(diff)

        # Also compute relative distance
        rel_dist = dist / np.linalg.norm(M1)

        return dist, rel_dist
    except:
        return float('inf'), None


def compare_via_procrustes():
    """Compare plain and Z3 tensors via Procrustes analysis."""
    print("=" * 80)
    print("Procrustes Analysis: Are Plain and Z3 Equivalent Under Basis Change?")
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

    # Initial tensors
    A_plain_0 = get_initial_tensor_potts_relT(pars_plain)
    A_z3_0 = get_initial_tensor_potts_relT(pars_z3)

    arr_plain_0 = A_plain_0.to_ndarray()
    arr_z3_0 = A_z3_0.to_ndarray()

    # Test initial tensors
    print("\n--- Initial Tensors (Step 0) ---")
    dist, rel_dist = procrustes_distance(arr_plain_0, arr_z3_0)
    print(f"  Direct comparison ||plain - Z3||/||plain||: {np.linalg.norm(arr_plain_0 - arr_z3_0)/np.linalg.norm(arr_plain_0):.2e}")
    print(f"  Procrustes distance (min over U): {dist:.2e}")
    print(f"  Relative Procrustes distance: {rel_dist:.2e}")

    # Now with GILT
    print("\n--- After GILT-TNR Step 1 ---")

    A_plain_1, _ = gilttnr_step(A_plain_0, 0.0, pars_plain)
    A_z3_1, _ = gilttnr_step(A_z3_0, 0.0, pars_z3)

    arr_plain_1 = A_plain_1.to_ndarray()
    arr_z3_1 = A_z3_1.to_ndarray()

    print(f"  Plain shape: {arr_plain_1.shape}")
    print(f"  Z3 shape: {arr_z3_1.shape}")

    if arr_plain_1.shape == arr_z3_1.shape:
        dist, rel_dist = procrustes_distance(arr_plain_1, arr_z3_1)
        print(f"  Direct comparison ||plain - Z3||/||plain||: {np.linalg.norm(arr_plain_1 - arr_z3_1)/np.linalg.norm(arr_plain_1):.2e}")
        print(f"  Procrustes distance: {dist:.2e}")
        print(f"  Relative Procrustes distance: {rel_dist:.2e}")
    else:
        print("  Shapes differ - cannot directly compare")
        # Try padding smaller tensor
        max_shape = tuple(max(s1, s2) for s1, s2 in zip(arr_plain_1.shape, arr_z3_1.shape))
        arr_plain_pad = np.zeros(max_shape, dtype=arr_plain_1.dtype)
        arr_z3_pad = np.zeros(max_shape, dtype=arr_z3_1.dtype)
        arr_plain_pad[:arr_plain_1.shape[0], :arr_plain_1.shape[1], :arr_plain_1.shape[2], :arr_plain_1.shape[3]] = arr_plain_1
        arr_z3_pad[:arr_z3_1.shape[0], :arr_z3_1.shape[1], :arr_z3_1.shape[2], :arr_z3_1.shape[3]] = arr_z3_1
        dist, rel_dist = procrustes_distance(arr_plain_pad, arr_z3_pad)
        print(f"  Padded Procrustes distance: {dist:.2e}")
        print(f"  Padded relative distance: {rel_dist:.2e}")

    # Also compare WITHOUT GILT
    print("\n--- After TRG Step 1 (NO GILT) ---")

    pars_no_gilt = dict(pars_plain)
    pars_no_gilt['gilt_eps'] = 0
    pars_no_gilt_z3 = dict(pars_z3)
    pars_no_gilt_z3['gilt_eps'] = 0

    A_plain_1_no = gilttnr_step(A_plain_0, 0.0, pars_no_gilt)[0]
    A_z3_1_no = gilttnr_step(A_z3_0, 0.0, pars_no_gilt_z3)[0]

    arr_plain_1_no = A_plain_1_no.to_ndarray()
    arr_z3_1_no = A_z3_1_no.to_ndarray()

    print(f"  Plain shape: {arr_plain_1_no.shape}")
    print(f"  Z3 shape: {arr_z3_1_no.shape}")

    if arr_plain_1_no.shape == arr_z3_1_no.shape:
        dist, rel_dist = procrustes_distance(arr_plain_1_no, arr_z3_1_no)
        print(f"  Direct comparison ||plain - Z3||/||plain||: {np.linalg.norm(arr_plain_1_no - arr_z3_1_no)/np.linalg.norm(arr_plain_1_no):.2e}")
        print(f"  Procrustes distance: {dist:.2e}")
        print(f"  Relative Procrustes distance: {rel_dist:.2e}")
    else:
        max_shape = tuple(max(s1, s2) for s1, s2 in zip(arr_plain_1_no.shape, arr_z3_1_no.shape))
        arr_plain_pad = np.zeros(max_shape, dtype=arr_plain_1_no.dtype)
        arr_z3_pad = np.zeros(max_shape, dtype=arr_z3_1_no.dtype)
        arr_plain_pad[:arr_plain_1_no.shape[0], :arr_plain_1_no.shape[1], :arr_plain_1_no.shape[2], :arr_plain_1_no.shape[3]] = arr_plain_1_no
        arr_z3_pad[:arr_z3_1_no.shape[0], :arr_z3_1_no.shape[1], :arr_z3_1_no.shape[2], :arr_z3_1_no.shape[3]] = arr_z3_1_no
        dist, rel_dist = procrustes_distance(arr_plain_pad, arr_z3_pad)
        print(f"  Padded Procrustes distance: {dist:.2e}")
        print(f"  Padded relative distance: {rel_dist:.2e}")

    # Continue for more steps
    print("\n--- Evolution Over Multiple Steps ---")
    print(f"{'Step':<6} {'GILT: rel_dist':<20} {'No GILT: rel_dist':<20}")
    print("-" * 50)

    A_p_gilt = A_plain_0
    A_z_gilt = A_z3_0
    A_p_no = A_plain_0
    A_z_no = A_z3_0

    for step in range(1, 5):
        A_p_gilt, _ = gilttnr_step(A_p_gilt, 0.0, pars_plain)
        A_z_gilt, _ = gilttnr_step(A_z_gilt, 0.0, pars_z3)
        A_p_no, _ = gilttnr_step(A_p_no, 0.0, pars_no_gilt)
        A_z_no, _ = gilttnr_step(A_z_no, 0.0, pars_no_gilt_z3)

        arr_p_gilt = A_p_gilt.to_ndarray()
        arr_z_gilt = A_z_gilt.to_ndarray()
        arr_p_no = A_p_no.to_ndarray()
        arr_z_no = A_z_no.to_ndarray()

        # Compute Procrustes distances
        if arr_p_gilt.shape == arr_z_gilt.shape:
            _, rel_gilt = procrustes_distance(arr_p_gilt, arr_z_gilt)
        else:
            rel_gilt = float('inf')

        if arr_p_no.shape == arr_z_no.shape:
            _, rel_no = procrustes_distance(arr_p_no, arr_z_no)
        else:
            rel_no = float('inf')

        gilt_str = f"{rel_gilt:.4e}" if rel_gilt < float('inf') else "shape mismatch"
        no_str = f"{rel_no:.4e}" if rel_no < float('inf') else "shape mismatch"
        print(f"{step:<6} {gilt_str:<20} {no_str:<20}")

    print("\n" + "=" * 80)
    print("Interpretation")
    print("=" * 80)
    print("""
If Procrustes distance stays small:
  → Tensors are equivalent up to basis change
  → Both methods compute the same physics

If Procrustes distance grows with GILT but stays small without GILT:
  → GILT is introducing a genuine difference
  → The filtering acts differently on the two representations

Key: Initial Procrustes distance should be ~0 (same tensor, different basis)
     After RG steps, this should remain small if both methods are consistent.
""")


if __name__ == "__main__":
    compare_via_procrustes()
