#!/usr/bin/env python3
"""
Debug the GILT filtering step specifically.

We've shown that pure TRG works correctly. The issue must be in GILT.
Let's trace through GILT step by step.
"""

import sys
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr/GiltTNR')
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr')

import numpy as np
from GiltTNR2D_Potts import get_initial_tensor_potts, get_initial_tensor_potts_z3
from GiltTNR2D import gilt_plaq, apply_gilt, get_envspec, build_Rp, optimize_Rp

def build_dft_matrix(N):
    """Build N×N DFT matrix."""
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega**(s*q) for q in range(N)] for s in range(N)]) / np.sqrt(N)
    return F

def transform_charge_to_spin(arr_charge, N=3):
    """Transform tensor from charge basis to spin basis."""
    F = build_dft_matrix(N)
    Fdag = F.conj().T
    return np.einsum('ia,jb,abcd,ck,dl->ijkl', Fdag, Fdag, arr_charge, F, F)

def test_get_envspec():
    """Test the get_envspec function (environment spectrum)."""
    print("="*70)
    print("Testing get_envspec (environment spectrum)")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))

    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    pars = {"gilt_eps": 1e-7, "verbosity": 0}

    print("\nComparing environment spectrum for each leg:")

    for leg in ['S', 'N', 'E', 'W']:
        print(f"\n  Leg {leg}:")

        U_z3, S_z3 = get_envspec(A_z3, A_z3, pars, where=leg)
        U_plain, S_plain = get_envspec(A_plain, A_plain, pars, where=leg)

        # Convert to numpy arrays for comparison
        S_z3_arr = S_z3.to_ndarray().flatten() if hasattr(S_z3, 'to_ndarray') else np.array(S_z3)
        S_plain_arr = S_plain.to_ndarray().flatten() if hasattr(S_plain, 'to_ndarray') else np.array(S_plain)

        S_z3_sorted = np.sort(np.abs(S_z3_arr))[::-1]
        S_plain_sorted = np.sort(np.abs(S_plain_arr))[::-1]

        print(f"    Z3: {len(S_z3_sorted)} singular values")
        print(f"    Plain: {len(S_plain_sorted)} singular values")
        print(f"    Z3 top 5 S: {S_z3_sorted[:5]}")
        print(f"    Plain top 5 S: {S_plain_sorted[:5]}")

        min_len = min(len(S_z3_sorted), len(S_plain_sorted))
        diff = np.linalg.norm(S_z3_sorted[:min_len] - S_plain_sorted[:min_len])
        rel_diff = diff / np.linalg.norm(S_plain_sorted[:min_len]) if np.linalg.norm(S_plain_sorted[:min_len]) > 0 else 0
        print(f"    Relative difference in S: {rel_diff:.2e}")

def test_apply_gilt():
    """Test the full GILT application."""
    print("\n" + "="*70)
    print("Testing apply_gilt")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))

    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    pars = {"gilt_eps": 1e-7, "verbosity": 0}

    # Test for each leg direction
    for leg in ['S']:  # Just test one leg for now
        print(f"\nApplying GILT to leg {leg}:")

        print("  Z3...")
        A1_z3, A2_z3, done_z3, err_z3 = apply_gilt(A_z3, A_z3, pars, where=leg)

        print("  Plain...")
        A1_plain, A2_plain, done_plain, err_plain = apply_gilt(A_plain, A_plain, pars, where=leg)

        print(f"  Z3: done={done_z3}, err={err_z3:.2e}, A1 shape={A1_z3.shape}")
        print(f"  Plain: done={done_plain}, err={err_plain:.2e}, A1 shape={A1_plain.shape}")

        # Compare the resulting tensors
        arr_z3 = A1_z3.to_ndarray()
        arr_plain = A1_plain.to_ndarray()

        # Transform Z3 to spin basis
        N = 3
        arr_z3_spin = transform_charge_to_spin(arr_z3, N)

        diff = np.linalg.norm(arr_z3_spin - arr_plain) / np.linalg.norm(arr_plain)
        print(f"  Tensor difference after apply_gilt (leg {leg}): {diff:.2e}")

    # Also test gilt_plaq (full plaquette GILT)
    print("\n" + "-"*50)
    print("Testing gilt_plaq (all 4 legs):")
    print("-"*50)

    A1_z3, A2_z3 = gilt_plaq(A_z3, A_z3, pars)
    A1_plain, A2_plain = gilt_plaq(A_plain, A_plain, pars)

    print(f"  Z3 A1 shape: {A1_z3.shape}")
    print(f"  Plain A1 shape: {A1_plain.shape}")

    A_z3_gilt = A1_z3
    A_plain_gilt = A1_plain

    print(f"\nResults:")
    print(f"  Z3 after GILT: shape = {A_z3_gilt.shape}")
    print(f"  Plain after GILT: shape = {A_plain_gilt.shape}")

    # Compare
    arr_z3 = A_z3_gilt.to_ndarray()
    arr_plain = A_plain_gilt.to_ndarray()

    # Transform Z3 to spin basis for comparison
    N = 3
    arr_z3_spin = transform_charge_to_spin(arr_z3, N)

    diff = np.linalg.norm(arr_z3_spin - arr_plain) / np.linalg.norm(arr_plain)
    print(f"\n  Relative difference (in spin basis): {diff:.2e}")

    # Compare singular values
    mat_z3 = arr_z3_spin.reshape(9, 9)
    mat_plain = arr_plain.reshape(9, 9)

    _, S_z3, _ = np.linalg.svd(mat_z3)
    _, S_plain, _ = np.linalg.svd(mat_plain)

    S_z3_sorted = np.sort(S_z3)[::-1]
    S_plain_sorted = np.sort(S_plain)[::-1]

    S_diff = np.linalg.norm(S_z3_sorted - S_plain_sorted) / np.linalg.norm(S_plain_sorted)
    print(f"  Singular value difference: {S_diff:.2e}")

    print(f"\n  Z3 top 5 S: {S_z3_sorted[:5]}")
    print(f"  Plain top 5 S: {S_plain_sorted[:5]}")

def test_gilt_vs_no_gilt():
    """Compare GILT-TNR step with and without GILT filtering."""
    print("\n" + "="*70)
    print("Comparing with and without GILT filtering")
    print("="*70)

    from GiltTNR2D import gilttnr_step, trg

    beta_c = np.log(1 + np.sqrt(3))

    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    pars_z3 = {"gilt_eps": 1e-7, "cg_chis": list(range(2, 25)), "cg_eps": 1e-10, "verbosity": 0, "symmetry_tensors": True}
    pars_plain = {"gilt_eps": 1e-7, "cg_chis": list(range(2, 25)), "cg_eps": 1e-10, "verbosity": 0, "symmetry_tensors": False}

    # GILT-TNR step (with GILT)
    print("\n1. With GILT filtering:")
    A_z3_gilt, _ = gilttnr_step(A_z3, 0.0, pars_z3)
    A_plain_gilt, _ = gilttnr_step(A_plain, 0.0, pars_plain)

    print(f"  Z3 shape: {A_z3_gilt.shape}, total dims: {[sum(d) for d in A_z3_gilt.shape]}")
    print(f"  Plain shape: {A_plain_gilt.shape}")

    # TRG step (no GILT)
    print("\n2. Without GILT filtering (pure TRG):")

    pars_trg_z3 = {"cg_chis": list(range(2, 25)), "cg_eps": 1e-10, "verbosity": 0, "symmetry_tensors": True}
    pars_trg_plain = {"cg_chis": list(range(2, 25)), "cg_eps": 1e-10, "verbosity": 0, "symmetry_tensors": False}

    A_z3_trg, _ = trg(A_z3, A_z3, 0.0, pars_trg_z3)
    A_plain_trg, _ = trg(A_plain, A_plain, 0.0, pars_trg_plain)

    print(f"  Z3 shape: {A_z3_trg.shape}, total dims: {[sum(d) for d in A_z3_trg.shape]}")
    print(f"  Plain shape: {A_plain_trg.shape}")

    # Compare dimensions
    print("\n3. Comparing final dimensions:")
    z3_gilt_dims = [sum(d) for d in A_z3_gilt.shape]
    z3_trg_dims = [sum(d) for d in A_z3_trg.shape]

    print(f"  GILT: Z3={z3_gilt_dims}, Plain={list(A_plain_gilt.shape)}")
    print(f"  TRG:  Z3={z3_trg_dims}, Plain={list(A_plain_trg.shape)}")

    if z3_gilt_dims != list(A_plain_gilt.shape):
        print("\n  WARNING: GILT gives different dimensions for Z3 vs Plain!")

if __name__ == "__main__":
    test_get_envspec()
    test_apply_gilt()
    test_gilt_vs_no_gilt()
