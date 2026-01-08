#!/usr/bin/env python3
"""
Debug the structure of U tensor in GILT for Z3.

U is the singular vectors of the environment matrix E.
U[chi, chi, n] where chi is the bond dimension and n indexes singular vectors.

The trace t[n] = Tr(U[:,:,n]) should identify "identity-like" modes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D import get_envspec
from GiltTNR2D_Potts import get_initial_tensor_potts_relT
from tensors import Tensor, TensorZ3

def main():
    print("=" * 70)
    print("Analyzing U Tensor Structure in GILT")
    print("=" * 70)

    pars_plain = {
        'q': 3,
        'relT': 1.0,
        'gilt_eps': 1e-6,
        'cg_chis': [16],
        'symmetry_tensors': False,
    }
    pars_z3 = dict(pars_plain)
    pars_z3['symmetry_tensors'] = True

    A_plain = get_initial_tensor_potts_relT(pars_plain)
    A_z3 = get_initial_tensor_potts_relT(pars_z3)

    # Get environment spectra
    U_plain, S_plain = get_envspec(A_plain, A_plain, pars_plain, where="S")
    U_z3, S_z3 = get_envspec(A_z3, A_z3, pars_z3, where="S")

    print("\n--- Plain Tensor ---")
    print(f"U shape: {U_plain.shape}")
    arr_U_plain = U_plain.to_ndarray()
    print(f"U array shape: {arr_U_plain.shape}")

    # For each singular vector, compute trace and check structure
    print(f"\nSingular values and traces:")
    S_plain_arr = S_plain.to_ndarray()
    for i in range(min(5, len(S_plain_arr))):
        U_i = arr_U_plain[:, :, i]
        trace_i = np.trace(U_i)
        frob_norm = np.linalg.norm(U_i, 'fro')
        print(f"  S[{i}]={S_plain_arr[i]:.4f}, Tr(U[{i}])={trace_i:.4f}, ||U[{i}]||_F={frob_norm:.4f}")

        # Check if U_i is identity-like
        # Max trace for normalized matrix is sqrt(chi)
        max_trace = np.sqrt(U_i.shape[0])
        identity_likeness = abs(trace_i) / max_trace
        print(f"    Identity-likeness: {identity_likeness:.4f} (1.0 = perfect identity)")

    print("\n--- Z3 Tensor ---")
    print(f"U shape: {U_z3.shape}")
    print(f"U qhape: {U_z3.qhape}")
    print(f"U dirs: {U_z3.dirs}")
    arr_U_z3 = U_z3.to_ndarray()
    print(f"U array shape: {arr_U_z3.shape}")

    S_z3_arr = S_z3.to_ndarray()
    print(f"\nSingular values and traces:")
    for i in range(min(5, len(S_z3_arr))):
        U_i = arr_U_z3[:, :, i]
        trace_i = np.trace(U_i)
        frob_norm = np.linalg.norm(U_i, 'fro')
        print(f"  S[{i}]={S_z3_arr[i]:.4f}, Tr(U[{i}])={trace_i:.4f}, ||U[{i}]||_F={frob_norm:.4f}")

        max_trace = np.sqrt(U_i.shape[0])
        identity_likeness = abs(trace_i) / max_trace
        print(f"    Identity-likeness: {identity_likeness:.4f}")

    # Check the actual U matrices
    print("\n--- Sample U matrices ---")
    print("\nPlain U[:,:,0] (largest singular value):")
    print(arr_U_plain[:, :, 0])

    print("\nZ3 U[:,:,0] (largest singular value):")
    print(arr_U_z3[:, :, 0])

    # Check charge structure of Z3 U
    print("\n--- Z3 U sectors ---")
    print(f"Number of sectors in U_z3: {len(U_z3.sects)}")
    for k, v in list(U_z3.sects.items())[:5]:
        print(f"  Key {k}: shape={v.shape}")

    # The key question: are the trace values computing correctly?
    # For Z3, the trace should be sum over diagonal blocks
    print("\n--- Manual trace check for Z3 ---")

    # Compute trace manually by summing diagonal elements
    manual_traces = []
    for i in range(arr_U_z3.shape[2]):
        U_i = arr_U_z3[:, :, i]
        tr = np.trace(U_i)
        manual_traces.append(tr)

    print(f"Manual traces (numpy): {manual_traces[:5]}")

    # Now compute using TensorZ3's trace
    t_z3 = ncon(U_z3, [1, 1, -1])
    t_z3_arr = t_z3.to_ndarray()
    print(f"ncon traces (Z3):      {t_z3_arr[:5]}")

    # Are they the same?
    if len(manual_traces) == len(t_z3_arr):
        diff = [abs(m - t) for m, t in zip(manual_traces, t_z3_arr)]
        print(f"Differences: {diff[:5]}")
        print(f"Max diff: {max(diff):.2e}")

    # Check the issue: Z3 trace might be summing over charge sectors differently
    print("\n--- Charge sector analysis ---")

    # For U with qhape [[0,1,2], [0,1,2], [0,1,2]]
    # Trace contracts indices 0 and 1
    # For charge conservation with dirs [-1, 1, -1]:
    #   -q0 + q1 - q2 = 0 mod 3
    #   q1 = q0 + q2 mod 3

    # So the trace sums over blocks where q0 = q1
    # But for dirs [-1, 1], the "dual" pairing is when -q0 + q1 = 0, i.e., q0 = q1

    print(f"U_z3 dirs for indices 0,1: {U_z3.dirs[:2]}")
    print("For trace, we need q0 and q1 to be 'dual'")
    print("With dirs [-1, 1]: -q0 + q1 = 0 mod 3, so q0 = q1")

    # Count which sectors contribute to trace
    trace_sectors = []
    for k in U_z3.sects:
        q0, q1, q2 = k
        if q0 == q1:  # These contribute to trace
            trace_sectors.append(k)
    print(f"\nSectors contributing to trace: {trace_sectors}")

if __name__ == "__main__":
    main()
