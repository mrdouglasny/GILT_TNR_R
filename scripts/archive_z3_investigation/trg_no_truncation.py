#!/usr/bin/env python3
"""
Test TRG step with NO truncation (very large chi) to verify that the
essential difference is ONLY in truncation, not in contractions.

If we don't truncate:
- Z3 tensor after TRG: shape [[9,9,9]] per leg (27 total)
- Plain tensor after TRG: shape (27,27,27,27)

They should be identical (in the same basis).
"""

import sys
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr/GiltTNR')
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr')

import numpy as np
from GiltTNR2D import trg
from GiltTNR2D_Potts import get_initial_tensor_potts, get_initial_tensor_potts_z3

def build_dft_matrix(N):
    """Build N×N DFT matrix."""
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega**(s*q) for q in range(N)] for s in range(N)]) / np.sqrt(N)
    return F

def transform_charge_to_spin(arr_charge, N=3):
    """Transform tensor from charge basis to spin basis."""
    chi = arr_charge.shape[0]

    if chi == N:
        F = build_dft_matrix(N)
        Fdag = F.conj().T
        return np.einsum('ia,jb,abcd,ck,dl->ijkl', Fdag, Fdag, arr_charge, F, F)

    # For larger tensors, need to build block transformation
    # But actually, for testing we can just compare singular values which are basis-independent
    return None

def test_no_truncation():
    """Test TRG with no truncation."""
    print("="*70)
    print("Testing TRG with no truncation")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))
    N = 3

    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    # Use very large chi to avoid any truncation
    max_chi = 100  # Way more than 3*3 = 9 needed
    pars_z3 = {"cg_chis": list(range(2, max_chi)), "cg_eps": 1e-15, "symmetry_tensors": True, "verbosity": 0}
    pars_plain = {"cg_chis": list(range(2, max_chi)), "cg_eps": 1e-15, "symmetry_tensors": False, "verbosity": 0}

    print("\nInitial tensors:")
    print(f"  Z3: shape = {A_z3.shape}")
    print(f"  Plain: shape = {A_plain.shape}")

    # Verify initial tensors match in spin basis
    arr_z3_init = A_z3.to_ndarray()
    arr_plain_init = A_plain.to_ndarray()
    arr_z3_spin_init = transform_charge_to_spin(arr_z3_init, N)

    init_diff = np.linalg.norm(arr_z3_spin_init - arr_plain_init) / np.linalg.norm(arr_plain_init)
    print(f"  Initial difference (spin basis): {init_diff:.2e}")

    # Run TRG
    print("\nRunning TRG (no truncation)...")
    A_z3_trg, _ = trg(A_z3, A_z3, 0.0, pars_z3)
    A_plain_trg, _ = trg(A_plain, A_plain, 0.0, pars_plain)

    print(f"\nAfter TRG:")
    print(f"  Z3: shape = {A_z3_trg.shape}")
    print(f"  Plain: shape = {A_plain_trg.shape}")

    arr_z3_trg = A_z3_trg.to_ndarray()
    arr_plain_trg = A_plain_trg.to_ndarray()

    print(f"  Z3 dense: shape = {arr_z3_trg.shape}")
    print(f"  Plain dense: shape = {arr_plain_trg.shape}")

    # Check shapes
    if arr_z3_trg.shape != arr_plain_trg.shape:
        print("\n  Shapes differ! This means truncation happened differently.")
        print("  Z3 kept fewer dimensions - this IS the bug!")

        # Look at the shape structure
        print(f"\n  Z3 shape structure: {A_z3_trg.shape}")
        # [[d0,d1,d2],...] means d0+d1+d2 total dims per leg

        return

    # Compare singular values (basis-independent)
    print("\nComparing singular values (basis-independent):")
    mat_z3 = arr_z3_trg.reshape(-1, arr_z3_trg.shape[0]**2)
    mat_plain = arr_plain_trg.reshape(-1, arr_plain_trg.shape[0]**2)

    # Handle non-square case
    if mat_z3.shape[0] != mat_z3.shape[1]:
        mat_z3 = arr_z3_trg.reshape(arr_z3_trg.shape[0]**2, arr_z3_trg.shape[0]**2)
        mat_plain = arr_plain_trg.reshape(arr_plain_trg.shape[0]**2, arr_plain_trg.shape[0]**2)

    _, S_z3, _ = np.linalg.svd(mat_z3)
    _, S_plain, _ = np.linalg.svd(mat_plain)

    S_z3_sorted = np.sort(S_z3)[::-1]
    S_plain_sorted = np.sort(S_plain)[::-1]

    min_len = min(len(S_z3_sorted), len(S_plain_sorted))
    S_diff = np.linalg.norm(S_z3_sorted[:min_len] - S_plain_sorted[:min_len]) / np.linalg.norm(S_plain_sorted[:min_len])

    print(f"  Relative difference in singular values: {S_diff:.2e}")

    if S_diff < 1e-10:
        print("\n✓ Tensors are identical (in appropriate bases) - truncation is the only difference!")
    else:
        print("\n✗ Tensors differ even without truncation - something else is wrong!")

def test_truncation_comparison():
    """Test that truncation itself is where divergence occurs."""
    print("\n" + "="*70)
    print("Testing truncation behavior")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))
    N = 3

    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    # Test various chi values
    for chi in [6, 9, 12, 15]:
        pars_z3 = {"cg_chis": list(range(2, chi+1)), "cg_eps": 1e-10, "symmetry_tensors": True, "verbosity": 0}
        pars_plain = {"cg_chis": list(range(2, chi+1)), "cg_eps": 1e-10, "symmetry_tensors": False, "verbosity": 0}

        A_z3_trg, _ = trg(A_z3, A_z3, 0.0, pars_z3)
        A_plain_trg, _ = trg(A_plain, A_plain, 0.0, pars_plain)

        # Get dimensions per leg
        z3_dims = [sum(d) for d in A_z3_trg.shape]
        plain_dims = list(A_plain_trg.shape)

        print(f"\nchi={chi}:")
        print(f"  Z3 dims per leg: {z3_dims}")
        print(f"  Plain dims per leg: {plain_dims}")

        # Compare singular values
        arr_z3 = A_z3_trg.to_ndarray()
        arr_plain = A_plain_trg.to_ndarray()

        # Pad smaller array to compare
        max_dim = max(arr_z3.shape[0], arr_plain.shape[0])
        if arr_z3.shape[0] != max_dim:
            arr_z3_pad = np.zeros((max_dim,)*4, dtype=arr_z3.dtype)
            arr_z3_pad[:arr_z3.shape[0], :arr_z3.shape[1], :arr_z3.shape[2], :arr_z3.shape[3]] = arr_z3
            arr_z3 = arr_z3_pad
        if arr_plain.shape[0] != max_dim:
            arr_plain_pad = np.zeros((max_dim,)*4, dtype=arr_plain.dtype)
            arr_plain_pad[:arr_plain.shape[0], :arr_plain.shape[1], :arr_plain.shape[2], :arr_plain.shape[3]] = arr_plain
            arr_plain = arr_plain_pad

        mat_z3 = arr_z3.reshape(max_dim**2, max_dim**2)
        mat_plain = arr_plain.reshape(max_dim**2, max_dim**2)

        _, S_z3, _ = np.linalg.svd(mat_z3)
        _, S_plain, _ = np.linalg.svd(mat_plain)

        # Only compare non-zero singular values
        S_z3_nz = S_z3[S_z3 > 1e-10]
        S_plain_nz = S_plain[S_plain > 1e-10]

        print(f"  Z3 non-zero S: {len(S_z3_nz)}, Plain non-zero S: {len(S_plain_nz)}")
        print(f"  Z3 top 5 S: {S_z3_nz[:5]}")
        print(f"  Plain top 5 S: {S_plain_nz[:5]}")

def analyze_truncation_choices():
    """Analyze how truncation choices differ."""
    print("\n" + "="*70)
    print("Analyzing truncation choices")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))
    N = 3

    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})

    # Look at the SVD singular values for TRG step
    arr = A_z3.to_ndarray()

    # TRG does SVD on legs [0,2] vs [1,3]
    mat = arr.transpose(0, 2, 1, 3).reshape(9, 9)

    _, S_full, _ = np.linalg.svd(mat)
    print(f"Full SVD singular values: {S_full}")
    print(f"Number of non-zero: {np.sum(S_full > 1e-10)}")

    # For Z3 truncation with chi=9:
    # The TensorZ3 will distribute dimensions across 3 sectors

    # Let's see what TensorZ3 actually does
    U, S, V = A_z3.svd([0, 2], [1, 3])
    S_arr = S.to_ndarray()

    print(f"\nTensorZ3 SVD S values: {S_arr.flatten()}")
    print(f"TensorZ3 S shape: {S.shape}")

    # Compare with full SVD
    S_z3_sorted = np.sort(np.abs(S_arr.flatten()))[::-1]
    S_full_sorted = np.sort(S_full)[::-1]

    print(f"\nComparing SVD singular values:")
    print(f"  Full (sorted): {S_full_sorted}")
    print(f"  Z3 (sorted): {S_z3_sorted}")

    diff = np.linalg.norm(S_z3_sorted - S_full_sorted) / np.linalg.norm(S_full_sorted)
    print(f"  Relative difference: {diff:.2e}")

if __name__ == "__main__":
    test_no_truncation()
    test_truncation_comparison()
    analyze_truncation_choices()
