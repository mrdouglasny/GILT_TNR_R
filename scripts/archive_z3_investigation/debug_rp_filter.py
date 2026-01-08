#!/usr/bin/env python3
"""
Debug the Rp filter computation and application.

Key findings so far:
1. Environment spectrum S matches perfectly
2. Singular values after GILT match perfectly
3. BUT tensor values differ by 110%!

This suggests the filter Rp is computed differently or applied differently.
"""

import sys
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr/GiltTNR')
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr')

import numpy as np
from GiltTNR2D_Potts import get_initial_tensor_potts, get_initial_tensor_potts_z3
from GiltTNR2D import get_envspec, optimize_Rp, apply_Rp, build_Rp

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

def test_rp_computation():
    """Test how Rp is computed."""
    print("="*70)
    print("Testing Rp computation")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))
    N = 3

    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    pars = {"gilt_eps": 1e-7, "verbosity": 0}

    leg = 'S'
    print(f"\nAnalyzing Rp for leg {leg}:")

    # Get environment spectrum
    U_z3, S_z3 = get_envspec(A_z3, A_z3, pars, where=leg)
    U_plain, S_plain = get_envspec(A_plain, A_plain, pars, where=leg)

    print("\n1. Environment spectrum:")
    S_z3_arr = S_z3.to_ndarray().flatten()
    S_plain_arr = S_plain.to_ndarray().flatten()
    print(f"   Z3 S: {np.sort(S_z3_arr)[::-1][:5]}")
    print(f"   Plain S: {np.sort(S_plain_arr)[::-1][:5]}")

    print("\n2. Eigenvector matrix U:")
    U_z3_arr = U_z3.to_ndarray() if hasattr(U_z3, 'to_ndarray') else np.array(U_z3)
    U_plain_arr = U_plain.to_ndarray() if hasattr(U_plain, 'to_ndarray') else np.array(U_plain)
    print(f"   Z3 U shape: {U_z3_arr.shape}")
    print(f"   Plain U shape: {U_plain_arr.shape}")

    # Check if U matrices are similar (up to basis transformation)
    # U_z3 should be in charge basis, U_plain in spin basis

    print("\n3. Computing Rp filter:")
    print(f"   Using gilt_eps = {pars['gilt_eps']}")

    # optimize_Rp computes the GILT filter
    Rp_z3, tp_z3 = optimize_Rp(U_z3, S_z3, pars)
    Rp_plain, tp_plain = optimize_Rp(U_plain, S_plain, pars)

    print(f"   Z3: tp={tp_z3}")
    print(f"   Plain: tp={tp_plain}")

    # Get Rp as array
    Rp_z3_arr = Rp_z3.to_ndarray() if hasattr(Rp_z3, 'to_ndarray') else np.array(Rp_z3)
    Rp_plain_arr = Rp_plain.to_ndarray() if hasattr(Rp_plain, 'to_ndarray') else np.array(Rp_plain)

    print(f"\n   Z3 Rp shape: {Rp_z3_arr.shape}")
    print(f"   Plain Rp shape: {Rp_plain_arr.shape}")

    print(f"\n   Z3 Rp:\n{Rp_z3_arr}")
    print(f"\n   Plain Rp:\n{Rp_plain_arr}")

    # Check if Rp matrices are similar
    if Rp_z3_arr.shape == Rp_plain_arr.shape:
        # Transform Z3 Rp to spin basis
        F = build_dft_matrix(N)
        Fdag = F.conj().T

        # Rp is a 2-leg matrix: Rp[a, b] where a,b are bond indices
        # Transform: Rp_spin = F† @ Rp_charge @ F
        Rp_z3_spin = Fdag @ Rp_z3_arr @ F

        diff = np.linalg.norm(Rp_z3_spin - Rp_plain_arr) / np.linalg.norm(Rp_plain_arr)
        print(f"\n   Rp_z3 transformed to spin vs Rp_plain: {diff:.2e}")

        print(f"\n   Rp_z3 in spin basis:\n{Rp_z3_spin}")

    print("\n4. Testing apply_Rp:")

    # Apply Rp to tensor
    # For leg 'S': A1 = ncon((A1, Rp1), ([-1,-2,3,-4], [3,-3]))
    #              A2 = ncon((A2, Rp2), ([1,-2,-3,-4], [-1,1]))

    A1_z3, A2_z3 = apply_Rp(A_z3, A_z3, Rp_z3, Rp_z3, where=leg)
    A1_plain, A2_plain = apply_Rp(A_plain, A_plain, Rp_plain, Rp_plain, where=leg)

    print(f"   A1_z3 shape: {A1_z3.shape}")
    print(f"   A1_plain shape: {A1_plain.shape}")

    # Compare
    arr_z3 = A1_z3.to_ndarray()
    arr_plain = A1_plain.to_ndarray()
    arr_z3_spin = transform_charge_to_spin(arr_z3, N)

    diff = np.linalg.norm(arr_z3_spin - arr_plain) / np.linalg.norm(arr_plain)
    print(f"\n   Tensor difference after apply_Rp: {diff:.2e}")

    # Compare singular values
    mat_z3 = arr_z3_spin.reshape(9, 9)
    mat_plain = arr_plain.reshape(9, 9)
    _, S_z3_new, _ = np.linalg.svd(mat_z3)
    _, S_plain_new, _ = np.linalg.svd(mat_plain)

    S_diff = np.linalg.norm(np.sort(S_z3_new)[::-1] - np.sort(S_plain_new)[::-1]) / np.linalg.norm(S_plain_new)
    print(f"   Singular value difference after apply_Rp: {S_diff:.2e}")

def test_identity_rp():
    """Test with identity Rp to verify apply_Rp works correctly."""
    print("\n" + "="*70)
    print("Testing with identity Rp")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))
    N = 3

    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    # Create identity Rp
    from tensors.symmetrytensors import TensorZ3
    from tensors.tensor import Tensor

    Rp_plain = Tensor.from_ndarray(np.eye(N, dtype=complex))

    # For Z3, identity needs to be in charge basis
    Rp_z3 = TensorZ3.from_ndarray(np.eye(N, dtype=complex), shape=[[1,1,1], [1,1,1]], dirs=[1,-1])

    print("Created identity Rp matrices")
    print(f"  Z3 Rp shape: {Rp_z3.shape}")
    print(f"  Plain Rp shape: {Rp_plain.shape}")

    leg = 'S'
    A1_z3, A2_z3 = apply_Rp(A_z3, A_z3, Rp_z3, Rp_z3, where=leg)
    A1_plain, A2_plain = apply_Rp(A_plain, A_plain, Rp_plain, Rp_plain, where=leg)

    # Compare with original
    arr_z3_orig = A_z3.to_ndarray()
    arr_z3_new = A1_z3.to_ndarray()
    arr_plain_orig = A_plain.to_ndarray()
    arr_plain_new = A1_plain.to_ndarray()

    diff_z3 = np.linalg.norm(arr_z3_new - arr_z3_orig) / np.linalg.norm(arr_z3_orig)
    diff_plain = np.linalg.norm(arr_plain_new - arr_plain_orig) / np.linalg.norm(arr_plain_orig)

    print(f"\nAfter applying identity Rp:")
    print(f"  Z3: ||A_new - A_orig|| / ||A_orig|| = {diff_z3:.2e}")
    print(f"  Plain: ||A_new - A_orig|| / ||A_orig|| = {diff_plain:.2e}")

if __name__ == "__main__":
    test_rp_computation()
    test_identity_rp()
