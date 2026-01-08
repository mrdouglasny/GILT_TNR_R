#!/usr/bin/env python3
"""
Check if the trace computation is the issue.

GILT uses trace t = Tr(U) to detect CDL modes.
If U is in charge basis vs spin basis, the trace should be different.
"""

import sys
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr/GiltTNR')
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr')

import numpy as np
from GiltTNR2D_Potts import get_initial_tensor_potts, get_initial_tensor_potts_z3
from GiltTNR2D import get_envspec

def build_dft_matrix(N):
    """Build N×N DFT matrix."""
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega**(s*q) for q in range(N)] for s in range(N)]) / np.sqrt(N)
    return F

def compute_trace_manually(U, N=3):
    """
    Compute trace of U manually.

    U has shape (chi, chi, D) where D is number of singular vectors.
    For each singular vector d, trace is U[i,i,d] summed over i.
    """
    U_arr = U.to_ndarray() if hasattr(U, 'to_ndarray') else np.array(U)

    chi = U_arr.shape[0]
    n_vecs = U_arr.shape[2]

    traces = []
    for d in range(n_vecs):
        t = 0
        for i in range(chi):
            t += U_arr[i, i, d]
        traces.append(t)

    return np.array(traces)

def test_trace_in_both_bases():
    """Test trace computation."""
    print("="*70)
    print("Testing trace computation in both bases")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))
    N = 3

    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    pars = {"gilt_eps": 1e-7, "verbosity": 0}

    leg = 'S'
    print(f"\nLeg {leg}:")

    # Get envspec
    U_z3, S_z3 = get_envspec(A_z3, A_z3, pars, where=leg)
    U_plain, S_plain = get_envspec(A_plain, A_plain, pars, where=leg)

    # Get U arrays
    U_z3_arr = U_z3.to_ndarray()
    U_plain_arr = U_plain.to_ndarray()

    print(f"  U_z3 shape: {U_z3_arr.shape}")
    print(f"  U_plain shape: {U_plain_arr.shape}")

    # Compute traces manually
    t_z3 = compute_trace_manually(U_z3)
    t_plain = compute_trace_manually(U_plain)

    print(f"\n  Traces (sorted by magnitude):")
    idx_z3 = np.argsort(-np.abs(t_z3))
    idx_plain = np.argsort(-np.abs(t_plain))

    print(f"  Z3 |t|: {np.abs(t_z3[idx_z3][:5])}")
    print(f"  Plain |t|: {np.abs(t_plain[idx_plain][:5])}")

    print(f"\n  Z3 t (complex): {t_z3[idx_z3][:5]}")
    print(f"  Plain t (complex): {t_plain[idx_plain][:5]}")

    # Transform U_z3 to spin basis and recompute trace
    print("\n  Transforming U_z3 to spin basis:")
    F = build_dft_matrix(N)
    Fdag = F.conj().T

    # U has shape (chi, chi, D) = (3, 3, 9)
    # Transform: U_spin[a, b, d] = Σ_{i,j} F†[a,i] U_charge[i,j,d] F[j,b]
    U_z3_spin = np.einsum('ai,ijd,jb->abd', Fdag, U_z3_arr, F)

    t_z3_spin = compute_trace_manually(U_z3_spin, N)

    print(f"  U_z3 in spin basis |t|: {np.abs(t_z3_spin[np.argsort(-np.abs(t_z3_spin))][:5])}")

    # Compare with plain
    diff_orig = np.linalg.norm(np.sort(np.abs(t_z3))[::-1] - np.sort(np.abs(t_plain))[::-1])
    diff_transformed = np.linalg.norm(np.sort(np.abs(t_z3_spin))[::-1] - np.sort(np.abs(t_plain))[::-1])

    print(f"\n  |t| difference (Z3 vs plain): {diff_orig:.2e}")
    print(f"  |t| difference (Z3->spin vs plain): {diff_transformed:.2e}")

if __name__ == "__main__":
    test_trace_in_both_bases()
