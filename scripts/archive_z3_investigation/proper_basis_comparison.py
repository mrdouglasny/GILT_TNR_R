#!/usr/bin/env python3
"""
Proper comparison: Transform Z3 tensors to spin basis before comparing.

The Z3 tensor in charge basis is related to plain tensor by:
  T_charge = (F⊗F⊗F†⊗F†) T_spin

To compare, we transform Z3 result back to spin basis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D import gilttnr_step
from GiltTNR2D_Potts import get_initial_tensor_potts_relT


def z3_to_spin_basis(arr_z3, shape_out=None):
    """
    Transform Z3 charge basis tensor to spin basis.

    T_spin[a,b,c,d] = sum_{i,j,k,l} F†[a,i] F†[b,j] F[c,k] F[d,l] T_charge[i,j,k,l]

    Note: Works only if arr_z3 shape is (3,3,3,3) on the physical indices.
    For truncated tensors, we need to handle the full bond structure.
    """
    # For the initial 3x3x3x3 tensor, this is straightforward
    if arr_z3.shape == (3, 3, 3, 3):
        omega = np.exp(2j * np.pi / 3)
        F = np.array([
            [1, 1, 1],
            [1, omega, omega**2],
            [1, omega**2, omega]
        ]) / np.sqrt(3)
        Fdag = F.T.conj()

        # Apply: T_spin = F† T_charge F (on each pair of legs)
        result = np.einsum('ai,bj,ijkl,ck,dl->abcd', Fdag, Fdag, arr_z3, F, F)
        return result

    # For truncated tensors with larger bond dimension, the transformation
    # is more complex because we don't know which charge sectors correspond
    # to which spin basis states.
    return None


def compare_tensors(arr1, arr2, name1, name2):
    """Compare two tensors after proper normalization."""
    if arr1 is None or arr2 is None:
        return None

    # Handle shape mismatch
    if arr1.shape != arr2.shape:
        print(f"  Shape mismatch: {name1}={arr1.shape}, {name2}={arr2.shape}")
        # Pad to same size
        max_shape = tuple(max(s1, s2) for s1, s2 in zip(arr1.shape, arr2.shape))
        arr1_pad = np.zeros(max_shape, dtype=arr1.dtype)
        arr2_pad = np.zeros(max_shape, dtype=arr2.dtype)
        slices1 = tuple(slice(0, s) for s in arr1.shape)
        slices2 = tuple(slice(0, s) for s in arr2.shape)
        arr1_pad[slices1] = arr1
        arr2_pad[slices2] = arr2
        arr1 = arr1_pad
        arr2 = arr2_pad

    # Normalize to same Frobenius norm
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)

    if norm1 > 1e-10 and norm2 > 1e-10:
        arr1_n = arr1 / norm1
        arr2_n = arr2 / norm2

        diff = np.linalg.norm(arr1_n - arr2_n)
        print(f"  ||{name1} - {name2}||/||{name1}|| (after normalization): {diff:.4e}")
        return diff

    return None


def main():
    print("=" * 80)
    print("Proper Basis Comparison: Transform Z3 to Spin Basis")
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

    pars_no_gilt = dict(pars_plain)
    pars_no_gilt['gilt_eps'] = 0
    pars_no_gilt_z3 = dict(pars_z3)
    pars_no_gilt_z3['gilt_eps'] = 0

    # Initial tensors
    A_plain_0 = get_initial_tensor_potts_relT(pars_plain)
    A_z3_0 = get_initial_tensor_potts_relT(pars_z3)

    arr_plain_0 = A_plain_0.to_ndarray()
    arr_z3_0 = A_z3_0.to_ndarray()

    # Transform Z3 to spin basis
    arr_z3_spin_0 = z3_to_spin_basis(arr_z3_0)

    print("\n--- Initial Tensors (Step 0) ---")
    print(f"  Plain shape: {arr_plain_0.shape}")
    print(f"  Z3 shape: {arr_z3_0.shape}")
    compare_tensors(arr_plain_0, arr_z3_spin_0, "plain", "Z3→spin")

    # After one step
    print("\n--- After Step 1 ---")

    # With GILT
    A_plain_gilt, _ = gilttnr_step(A_plain_0, 0.0, pars_plain)
    A_z3_gilt, _ = gilttnr_step(A_z3_0, 0.0, pars_z3)

    # Without GILT
    A_plain_no, _ = gilttnr_step(A_plain_0, 0.0, pars_no_gilt)
    A_z3_no, _ = gilttnr_step(A_z3_0, 0.0, pars_no_gilt_z3)

    arr_plain_gilt = A_plain_gilt.to_ndarray()
    arr_z3_gilt = A_z3_gilt.to_ndarray()
    arr_plain_no = A_plain_no.to_ndarray()
    arr_z3_no = A_z3_no.to_ndarray()

    print("\nWith GILT:")
    print(f"  Plain shape: {arr_plain_gilt.shape}")
    print(f"  Z3 shape: {arr_z3_gilt.shape}")

    print("\nWithout GILT:")
    print(f"  Plain shape: {arr_plain_no.shape}")
    print(f"  Z3 shape: {arr_z3_no.shape}")

    # Key insight: After truncation, we can't easily transform back because
    # the basis is mixed. Instead, compare physically observable quantities.

    print("\n" + "=" * 80)
    print("Comparing Physical Quantities (Invariant Under Basis)")
    print("=" * 80)

    print("\n--- Tensor Norms (scale-dependent but basis-independent) ---")
    print(f"  GILT: plain={np.linalg.norm(arr_plain_gilt):.4e}, Z3={np.linalg.norm(arr_z3_gilt):.4e}")
    print(f"  No GILT: plain={np.linalg.norm(arr_plain_no):.4e}, Z3={np.linalg.norm(arr_z3_no):.4e}")

    # Singular value spectrum (basis independent)
    print("\n--- Singular Value Spectra (reshape to matrix) ---")

    def get_svd_spectrum(arr):
        """Get singular values of tensor reshaped as matrix."""
        d1 = arr.shape[0] * arr.shape[1]
        d2 = arr.shape[2] * arr.shape[3]
        M = arr.reshape(d1, d2)
        s = np.linalg.svd(M, compute_uv=False)
        return s / s[0]  # Normalize by largest

    s_plain_gilt = get_svd_spectrum(arr_plain_gilt)
    s_z3_gilt = get_svd_spectrum(arr_z3_gilt)
    s_plain_no = get_svd_spectrum(arr_plain_no)
    s_z3_no = get_svd_spectrum(arr_z3_no)

    print("\nWith GILT (top 10 normalized singular values):")
    print(f"  Plain: {s_plain_gilt[:10]}")
    print(f"  Z3:    {s_z3_gilt[:10]}")

    print("\nWithout GILT:")
    print(f"  Plain: {s_plain_no[:10]}")
    print(f"  Z3:    {s_z3_no[:10]}")

    # Compare spectra
    min_len_gilt = min(len(s_plain_gilt), len(s_z3_gilt))
    min_len_no = min(len(s_plain_no), len(s_z3_no))

    diff_gilt = np.linalg.norm(s_plain_gilt[:min_len_gilt] - s_z3_gilt[:min_len_gilt])
    diff_no = np.linalg.norm(s_plain_no[:min_len_no] - s_z3_no[:min_len_no])

    print(f"\n||S_plain - S_z3|| (spectrum difference):")
    print(f"  With GILT: {diff_gilt:.4e}")
    print(f"  Without GILT: {diff_no:.4e}")

    # Evolution over steps
    print("\n" + "=" * 80)
    print("Evolution of Spectrum Difference Over Steps")
    print("=" * 80)

    A_p_gilt = A_plain_0
    A_z_gilt = A_z3_0
    A_p_no = A_plain_0
    A_z_no = A_z3_0

    print(f"\n{'Step':<6} {'GILT spec diff':<16} {'No GILT spec diff':<16} {'GILT norm ratio':<16} {'No GILT norm ratio':<16}")
    print("-" * 80)

    for step in range(1, 6):
        A_p_gilt, _ = gilttnr_step(A_p_gilt, 0.0, pars_plain)
        A_z_gilt, _ = gilttnr_step(A_z_gilt, 0.0, pars_z3)
        A_p_no, _ = gilttnr_step(A_p_no, 0.0, pars_no_gilt)
        A_z_no, _ = gilttnr_step(A_z_no, 0.0, pars_no_gilt_z3)

        arr_p_gilt = A_p_gilt.to_ndarray()
        arr_z_gilt = A_z_gilt.to_ndarray()
        arr_p_no = A_p_no.to_ndarray()
        arr_z_no = A_z_no.to_ndarray()

        # Singular value spectra
        s_p_gilt = get_svd_spectrum(arr_p_gilt)
        s_z_gilt = get_svd_spectrum(arr_z_gilt)
        s_p_no = get_svd_spectrum(arr_p_no)
        s_z_no = get_svd_spectrum(arr_z_no)

        min_len = min(len(s_p_gilt), len(s_z_gilt))
        diff_gilt = np.linalg.norm(s_p_gilt[:min_len] - s_z_gilt[:min_len])

        min_len = min(len(s_p_no), len(s_z_no))
        diff_no = np.linalg.norm(s_p_no[:min_len] - s_z_no[:min_len])

        # Norm ratio
        norm_ratio_gilt = np.linalg.norm(arr_z_gilt) / np.linalg.norm(arr_p_gilt)
        norm_ratio_no = np.linalg.norm(arr_z_no) / np.linalg.norm(arr_p_no)

        print(f"{step:<6} {diff_gilt:<16.4e} {diff_no:<16.4e} {norm_ratio_gilt:<16.4f} {norm_ratio_no:<16.4f}")

    print("\n" + "=" * 80)
    print("Conclusion")
    print("=" * 80)
    print("""
If spectrum difference stays small but grows with GILT:
  → GILT is changing the tensor differently for plain vs Z3
  → The filtering is not equivalent in the two representations

If norm ratio stays ~1 without GILT but diverges with GILT:
  → GILT filtering removes different amounts of information
""")


if __name__ == "__main__":
    main()
