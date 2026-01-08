#!/usr/bin/env python3
"""
Debug the GILT trace computation for Z3 tensors.

The GILT algorithm computes:
  t = ncon(U, [1,1,-1])  # Trace of reshaped singular vectors

For equivariant tensors, this must be done carefully.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D import get_envspec
from GiltTNR2D_Potts import get_initial_tensor_potts_relT
from tensors import Tensor, TensorZ3

def test_trace_for_z3():
    """Test the trace operation on Z3 tensors."""
    print("=" * 70)
    print("Testing Trace Operation for Z3")
    print("=" * 70)

    # Create a simple Z3 matrix tensor
    # Shape [[d0, d1, d2], [d0', d1', d2']]
    # For trace, indices must be "dual" - same qhape, opposite dirs

    # Simple case: 3x3 matrix with Z3 structure
    dim = [1, 1, 1]
    qim = [0, 1, 2]

    # Create matrix with dirs [1, -1] so trace makes sense
    M = TensorZ3([dim, dim], qhape=[qim, qim], dirs=[1, -1], charge=0)

    # Fill with identity-like values in charge-0 sectors
    # For dirs [1, -1]: charge = q0 - q1
    # Charge-0 blocks: (0,0), (1,1), (2,2)
    M.sects[(0, 0)] = np.array([[1.0]])
    M.sects[(1, 1)] = np.array([[1.0]])
    M.sects[(2, 2)] = np.array([[1.0]])

    print("\nZ3 matrix M (identity-like):")
    print(f"  shape: {M.shape}")
    print(f"  qhape: {M.qhape}")
    print(f"  dirs: {M.dirs}")
    print(f"  sectors: {list(M.sects.keys())}")

    arr_M = M.to_ndarray()
    print(f"\nAs array:\n{arr_M}")

    # Trace using numpy
    trace_numpy = np.trace(arr_M)
    print(f"\nnumpy trace: {trace_numpy}")

    # Trace using ncon
    try:
        trace_ncon = ncon((M,), ([1, 1]))
        print(f"ncon trace (tensor): {trace_ncon}")
        if hasattr(trace_ncon, 'to_ndarray'):
            print(f"ncon trace (value): {trace_ncon.to_ndarray()}")
    except Exception as e:
        print(f"ncon trace failed: {e}")

    # Now test with actual GILT environment spectrum
    print("\n" + "=" * 70)
    print("Testing with GILT Environment")
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

    print("\nGetting environment spectrum (South leg)...")

    # Plain tensor environment
    U_plain, S_plain = get_envspec(A_plain, A_plain, pars_plain, where="S")
    print(f"\nPlain U shape: {U_plain.shape}")
    print(f"Plain S shape: {S_plain.shape}")

    # Compute trace for plain
    t_plain = ncon(U_plain, [1, 1, -1])
    print(f"Plain t (trace) shape: {t_plain.shape}")
    t_plain_arr = t_plain.to_ndarray()
    print(f"Plain t values: {t_plain_arr[:5]}")

    # Z3 tensor environment
    U_z3, S_z3 = get_envspec(A_z3, A_z3, pars_z3, where="S")
    print(f"\nZ3 U shape: {U_z3.shape}")
    print(f"Z3 U qhape: {U_z3.qhape}")
    print(f"Z3 U dirs: {U_z3.dirs}")
    print(f"Z3 S shape: {S_z3.shape}")

    # Compute trace for Z3
    t_z3 = ncon(U_z3, [1, 1, -1])
    print(f"\nZ3 t (trace) shape: {t_z3.shape}")
    if hasattr(t_z3, 'qhape'):
        print(f"Z3 t qhape: {t_z3.qhape}")
    t_z3_arr = t_z3.to_ndarray()
    print(f"Z3 t values: {t_z3_arr[:5]}")

    # Compare
    print("\n--- Comparison ---")
    print(f"Plain t norm: {np.linalg.norm(t_plain_arr):.6f}")
    print(f"Z3 t norm:    {np.linalg.norm(t_z3_arr):.6f}")

    # Check if the trace values are similar
    if len(t_plain_arr) == len(t_z3_arr):
        diff = np.linalg.norm(t_plain_arr - t_z3_arr)
        print(f"||t_plain - t_z3||: {diff:.6e}")
    else:
        print(f"Sizes differ: plain={len(t_plain_arr)}, z3={len(t_z3_arr)}")


if __name__ == "__main__":
    test_trace_for_z3()
