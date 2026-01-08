#!/usr/bin/env python3
"""
Debug the Rp.split() operation which creates Rp1 and Rp2.

apply_gilt does:
1. Get envspec (U, S)
2. Normalize S
3. optimize_Rp -> Rp
4. Rp.split(0, 1, eps) -> Rp1, s, Rp2
5. apply_Rp(A1, A2, Rp1, Rp2)

We showed optimize_Rp gives matching Rp. The split might differ.
"""

import sys
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr/GiltTNR')
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr')

import numpy as np
from GiltTNR2D_Potts import get_initial_tensor_potts, get_initial_tensor_potts_z3
from GiltTNR2D import get_envspec, optimize_Rp, apply_Rp

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

def test_split():
    """Test the Rp.split() operation."""
    print("="*70)
    print("Testing Rp.split()")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))
    N = 3

    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    pars = {"gilt_eps": 1e-7, "verbosity": 0}
    leg = 'S'

    # Step 1: Get envspec
    print("\n1. Getting environment spectrum:")
    U_z3, S_z3 = get_envspec(A_z3, A_z3, pars, where=leg)
    U_plain, S_plain = get_envspec(A_plain, A_plain, pars, where=leg)

    # Step 2: Normalize S
    S_z3_norm = S_z3 / S_z3.sum()
    S_plain_norm = S_plain / S_plain.sum()

    print(f"   Z3 S (normalized, top 5): {np.sort(S_z3_norm.to_ndarray().flatten())[::-1][:5]}")
    print(f"   Plain S (normalized, top 5): {np.sort(S_plain_norm.to_ndarray().flatten())[::-1][:5]}")

    # Step 3: optimize_Rp
    print("\n2. Computing Rp via optimize_Rp:")
    Rp_z3, err_z3 = optimize_Rp(U_z3, S_z3_norm, pars)
    Rp_plain, err_plain = optimize_Rp(U_plain, S_plain_norm, pars)

    print(f"   Z3 Rp shape: {Rp_z3.shape}, insertion error: {err_z3:.2e}")
    print(f"   Plain Rp shape: {Rp_plain.shape}, insertion error: {err_plain:.2e}")

    # Show Rp
    Rp_z3_arr = Rp_z3.to_ndarray()
    Rp_plain_arr = Rp_plain.to_ndarray()
    print(f"\n   Z3 Rp:\n{Rp_z3_arr}")
    print(f"\n   Plain Rp:\n{Rp_plain_arr}")

    # Step 4: Split Rp
    print("\n3. Splitting Rp:")
    spliteps = pars["gilt_eps"] * 1e-3

    Rp1_z3, s_z3, Rp2_z3, spliterr_z3 = Rp_z3.split(0, 1, eps=spliteps, return_rel_err=True, return_sings=True)
    Rp1_plain, s_plain, Rp2_plain, spliterr_plain = Rp_plain.split(0, 1, eps=spliteps, return_rel_err=True, return_sings=True)

    print(f"   Split eps: {spliteps}")

    # s is the singular values from the split
    s_z3_arr = s_z3.to_ndarray().flatten() if hasattr(s_z3, 'to_ndarray') else np.array(s_z3)
    s_plain_arr = s_plain.to_ndarray().flatten() if hasattr(s_plain, 'to_ndarray') else np.array(s_plain)

    print(f"\n   Z3 split singular values s: {s_z3_arr}")
    print(f"   Plain split singular values s: {s_plain_arr}")

    print(f"\n   Z3 Rp1 shape: {Rp1_z3.shape}, Rp2 shape: {Rp2_z3.shape}")
    print(f"   Plain Rp1 shape: {Rp1_plain.shape}, Rp2 shape: {Rp2_plain.shape}")

    print(f"\n   Z3 split error: {spliterr_z3:.2e}")
    print(f"   Plain split error: {spliterr_plain:.2e}")

    # Show Rp1 and Rp2
    Rp1_z3_arr = Rp1_z3.to_ndarray()
    Rp2_z3_arr = Rp2_z3.to_ndarray()
    Rp1_plain_arr = Rp1_plain.to_ndarray()
    Rp2_plain_arr = Rp2_plain.to_ndarray()

    print(f"\n   Z3 Rp1:\n{Rp1_z3_arr}")
    print(f"   Plain Rp1:\n{Rp1_plain_arr}")

    print(f"\n   Z3 Rp2:\n{Rp2_z3_arr}")
    print(f"   Plain Rp2:\n{Rp2_plain_arr}")

    # Compare Rp1 and Rp2 in same basis
    F = build_dft_matrix(N)
    Fdag = F.conj().T

    if Rp1_z3_arr.shape == Rp1_plain_arr.shape:
        # Rp1 and Rp2 are 2-leg matrices
        # Rp1[old_leg, new_leg] transforms old to new
        # Rp2[new_leg, old_leg] transforms new to old

        # Transform Rp1_z3 to spin basis
        Rp1_z3_spin = Fdag @ Rp1_z3_arr  # transform first index
        # Actually need to think about the indices...

        print(f"\n   Direct comparison (may need basis transform):")
        print(f"   ||Rp1_z3 - Rp1_plain|| / ||Rp1_plain|| = {np.linalg.norm(Rp1_z3_arr - Rp1_plain_arr) / np.linalg.norm(Rp1_plain_arr):.2e}")

    # Step 5: Apply Rp1, Rp2
    print("\n4. Applying Rp1, Rp2:")
    A1_z3, A2_z3 = apply_Rp(A_z3, A_z3, Rp1_z3, Rp2_z3, where=leg)
    A1_plain, A2_plain = apply_Rp(A_plain, A_plain, Rp1_plain, Rp2_plain, where=leg)

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

if __name__ == "__main__":
    test_split()
