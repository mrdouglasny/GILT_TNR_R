#!/usr/bin/env python3
"""
Test GILT after one TRG step to see if the issue persists
when Rp is not identity.

After TRG, the tensor develops CDL (corner double line) structure
which GILT should filter. The Rp matrices should then be non-trivial.
"""

import sys
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr/GiltTNR')
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr')

import numpy as np
from GiltTNR2D_Potts import get_initial_tensor_potts, get_initial_tensor_potts_z3
from GiltTNR2D import trg, get_envspec, optimize_Rp, apply_Rp

def build_dft_matrix(N):
    """Build N×N DFT matrix."""
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega**(s*q) for q in range(N)] for s in range(N)]) / np.sqrt(N)
    return F

def transform_charge_to_spin_large(arr_charge, N=3):
    """Transform tensor from charge basis to spin basis for larger dims."""
    chi = arr_charge.shape[0]
    if chi == N:
        F = build_dft_matrix(N)
        Fdag = F.conj().T
        return np.einsum('ia,jb,abcd,ck,dl->ijkl', Fdag, Fdag, arr_charge, F, F)
    else:
        # For larger tensors after TRG
        block_size = chi // N
        F = build_dft_matrix(N)
        Fdag = F.conj().T
        # Build block-wise transformation
        F_large = np.kron(np.eye(block_size), F)
        Fdag_large = F_large.conj().T
        return np.einsum('ia,jb,abcd,ck,dl->ijkl', Fdag_large, Fdag_large, arr_charge, F_large, F_large)

def test_after_trg():
    """Test GILT after one TRG step."""
    print("="*70)
    print("Testing GILT after one TRG step")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))
    N = 3

    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    pars_trg_z3 = {"cg_chis": list(range(2, 25)), "cg_eps": 1e-10, "verbosity": 0, "symmetry_tensors": True}
    pars_trg_plain = {"cg_chis": list(range(2, 25)), "cg_eps": 1e-10, "verbosity": 0, "symmetry_tensors": False}

    # Do one TRG step (no GILT)
    print("\n1. Running one TRG step...")
    A_z3_trg, _ = trg(A_z3, A_z3, 0.0, pars_trg_z3)
    A_plain_trg, _ = trg(A_plain, A_plain, 0.0, pars_trg_plain)

    print(f"   Z3 shape after TRG: {A_z3_trg.shape}, total dims: {[sum(d) for d in A_z3_trg.shape]}")
    print(f"   Plain shape after TRG: {A_plain_trg.shape}")

    # Check if tensors are equivalent
    arr_z3 = A_z3_trg.to_ndarray()
    arr_plain = A_plain_trg.to_ndarray()

    # Compare singular values
    mat_z3 = arr_z3.reshape(81, 81)
    mat_plain = arr_plain.reshape(81, 81)
    _, S_z3, _ = np.linalg.svd(mat_z3)
    _, S_plain, _ = np.linalg.svd(mat_plain)

    S_diff = np.linalg.norm(np.sort(S_z3)[::-1] - np.sort(S_plain)[::-1]) / np.linalg.norm(S_plain)
    print(f"   Singular value difference after TRG: {S_diff:.2e}")

    # Now test GILT on the TRG output
    print("\n2. Testing GILT environment spectrum:")
    pars_gilt = {"gilt_eps": 1e-6, "verbosity": 0}

    for leg in ['S']:
        print(f"\n   Leg {leg}:")

        U_z3, S_z3 = get_envspec(A_z3_trg, A_z3_trg, pars_gilt, where=leg)
        U_plain, S_plain = get_envspec(A_plain_trg, A_plain_trg, pars_gilt, where=leg)

        S_z3_arr = S_z3.to_ndarray().flatten()
        S_plain_arr = S_plain.to_ndarray().flatten()

        S_z3_sorted = np.sort(np.abs(S_z3_arr))[::-1]
        S_plain_sorted = np.sort(np.abs(S_plain_arr))[::-1]

        print(f"   Z3 envspec S (top 10): {S_z3_sorted[:10]}")
        print(f"   Plain envspec S (top 10): {S_plain_sorted[:10]}")

        min_len = min(len(S_z3_sorted), len(S_plain_sorted))
        S_diff = np.linalg.norm(S_z3_sorted[:min_len] - S_plain_sorted[:min_len]) / np.linalg.norm(S_plain_sorted[:min_len])
        print(f"   Envspec S difference: {S_diff:.2e}")

        # Normalize and compute Rp
        S_z3_norm = S_z3 / S_z3.sum()
        S_plain_norm = S_plain / S_plain.sum()

        Rp_z3, err_z3 = optimize_Rp(U_z3, S_z3_norm, pars_gilt)
        Rp_plain, err_plain = optimize_Rp(U_plain, S_plain_norm, pars_gilt)

        print(f"\n   Z3 Rp insertion error: {err_z3:.2e}")
        print(f"   Plain Rp insertion error: {err_plain:.2e}")

        # Check if Rp is non-trivial
        Rp_z3_arr = Rp_z3.to_ndarray()
        Rp_plain_arr = Rp_plain.to_ndarray()

        Rp_z3_diag = np.diag(Rp_z3_arr)
        Rp_plain_diag = np.diag(Rp_plain_arr) if Rp_plain_arr.ndim == 2 else Rp_plain_arr

        print(f"\n   Z3 Rp diagonal: {Rp_z3_diag[:5]}...")
        print(f"   Plain Rp diagonal: {Rp_plain_diag[:5]}...")

        # Check off-diagonal
        Rp_z3_offdiag_norm = np.linalg.norm(Rp_z3_arr - np.diag(np.diag(Rp_z3_arr)))
        Rp_plain_offdiag_norm = np.linalg.norm(Rp_plain_arr - np.diag(np.diag(Rp_plain_arr)))

        print(f"\n   Z3 Rp off-diagonal norm: {Rp_z3_offdiag_norm:.2e}")
        print(f"   Plain Rp off-diagonal norm: {Rp_plain_offdiag_norm:.2e}")

if __name__ == "__main__":
    test_after_trg()
