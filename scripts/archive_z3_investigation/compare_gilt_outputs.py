#!/usr/bin/env python3
"""
Compare GILT outputs between plain and Z3 tensors by converting to same basis.

Strategy:
1. Start with identical tensors (plain and Z3 representation of same tensor)
2. Apply GILT to each
3. Convert Z3 result back to plain basis
4. Compare the two plain tensors
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D import gilttnr_step, get_envspec, optimize_Rp
from GiltTNR2D_Potts import get_initial_tensor_potts_relT

def z3_to_plain(arr_z3):
    """Transform Z3 charge basis tensor to plain spin basis."""
    omega = np.exp(2j * np.pi / 3)
    F = np.array([
        [1, 1, 1],
        [1, omega, omega**2],
        [1, omega**2, omega]
    ]) / np.sqrt(3)
    Fdag = F.T.conj()

    # T_spin[a,b,c,d] = sum_{i,j,k,l} F†[a,i] F†[b,j] F[c,k] F[d,l] T_charge[i,j,k,l]
    # Only transform the first 3 indices of each leg
    shape = arr_z3.shape
    result = arr_z3.copy()

    # Transform each leg
    # Leg 0: F†
    result = np.tensordot(Fdag, result, axes=([1], [0]))
    # Leg 1: F†
    result = np.tensordot(Fdag, result, axes=([1], [1]))
    result = np.moveaxis(result, 0, 1)
    # Leg 2: F
    result = np.tensordot(F, result, axes=([1], [2]))
    result = np.moveaxis(result, 0, 2)
    # Leg 3: F
    result = np.tensordot(F, result, axes=([1], [3]))
    result = np.moveaxis(result, 0, 3)

    return result


def compare_gilt_step():
    """Compare one full GILT-TNR step between plain and Z3."""
    print("=" * 80)
    print("Comparing GILT Outputs: Plain vs Z3 (converted to plain)")
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

    # Get initial tensors
    A_plain_0 = get_initial_tensor_potts_relT(pars_plain)
    A_z3_0 = get_initial_tensor_potts_relT(pars_z3)

    arr_plain_0 = A_plain_0.to_ndarray()
    arr_z3_0 = A_z3_0.to_ndarray()

    # Verify initial tensors are equivalent
    arr_z3_to_plain_0 = z3_to_plain(arr_z3_0)
    diff_0 = np.linalg.norm(arr_plain_0 - arr_z3_to_plain_0)
    print(f"\nInitial tensors:")
    print(f"  ||plain - Z3_to_plain||: {diff_0:.2e}")
    print(f"  (Should be ~0 if tensor construction is correct)")

    # Step 1: Apply one GILT-TNR step
    print("\n" + "-" * 80)
    print("After ONE GILT-TNR step:")
    print("-" * 80)

    A_plain_1, log_plain = gilttnr_step(A_plain_0, 0.0, pars_plain)
    A_z3_1, log_z3 = gilttnr_step(A_z3_0, 0.0, pars_z3)

    arr_plain_1 = A_plain_1.to_ndarray()
    arr_z3_1 = A_z3_1.to_ndarray()

    print(f"\nShapes: plain={arr_plain_1.shape}, Z3={arr_z3_1.shape}")
    print(f"Norms: plain={np.linalg.norm(arr_plain_1):.4e}, Z3={np.linalg.norm(arr_z3_1):.4e}")
    print(f"Log factors: plain={log_plain:.6f}, Z3={log_z3:.6f}")

    # Convert Z3 to plain for comparison
    # Note: After truncation, the shapes may differ!
    if arr_plain_1.shape == arr_z3_1.shape:
        arr_z3_to_plain_1 = z3_to_plain(arr_z3_1)
        diff_1 = np.linalg.norm(arr_plain_1 - arr_z3_to_plain_1)
        rel_diff_1 = diff_1 / np.linalg.norm(arr_plain_1)
        print(f"\n||plain_1 - Z3_1_to_plain||: {diff_1:.4e}")
        print(f"Relative difference: {rel_diff_1:.4%}")
    else:
        print(f"\nCannot directly compare: shapes differ")
        print(f"  plain: {arr_plain_1.shape}")
        print(f"  Z3:    {arr_z3_1.shape}")


def compare_gilt_environment():
    """Compare the GILT environment computation between plain and Z3."""
    print("\n" + "=" * 80)
    print("Comparing GILT Environment Computation")
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

    A_plain = get_initial_tensor_potts_relT(pars_plain)
    A_z3 = get_initial_tensor_potts_relT(pars_z3)

    # Get environment spectra
    U_plain, S_plain = get_envspec(A_plain, A_plain, pars_plain, where="S")
    U_z3, S_z3 = get_envspec(A_z3, A_z3, pars_z3, where="S")

    arr_U_plain = U_plain.to_ndarray()
    arr_U_z3 = U_z3.to_ndarray()

    S_plain_arr = S_plain.to_ndarray()
    S_z3_arr = S_z3.to_ndarray()

    print(f"\nU shapes: plain={arr_U_plain.shape}, Z3={arr_U_z3.shape}")
    print(f"S lengths: plain={len(S_plain_arr)}, Z3={len(S_z3_arr)}")

    # The singular VALUES should be the same (just reordered)
    S_plain_sorted = sorted(S_plain_arr, reverse=True)
    S_z3_sorted = sorted(S_z3_arr, reverse=True)

    print(f"\nSorted singular values comparison (should be identical):")
    min_len = min(len(S_plain_sorted), len(S_z3_sorted))
    for i in range(min(10, min_len)):
        diff = abs(S_plain_sorted[i] - S_z3_sorted[i])
        print(f"  S[{i}]: plain={S_plain_sorted[i]:.4e}, Z3={S_z3_sorted[i]:.4e}, diff={diff:.2e}")


def compare_gilt_filter():
    """Compare the GILT R' filter between plain and Z3."""
    print("\n" + "=" * 80)
    print("Comparing GILT R' Filter")
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

    # After one step (when truncation has happened)
    A_plain_0 = get_initial_tensor_potts_relT(pars_plain)
    A_z3_0 = get_initial_tensor_potts_relT(pars_z3)

    A_plain_1, _ = gilttnr_step(A_plain_0, 0.0, pars_plain)
    A_z3_1, _ = gilttnr_step(A_z3_0, 0.0, pars_z3)

    # Now get the Rp filter for step 2
    U_plain, S_plain = get_envspec(A_plain_1, A_plain_1, pars_plain, where="S")
    U_z3, S_z3 = get_envspec(A_z3_1, A_z3_1, pars_z3, where="S")

    Rp_plain, err_plain = optimize_Rp(U_plain, S_plain, pars_plain)
    Rp_z3, err_z3 = optimize_Rp(U_z3, S_z3, pars_z3)

    arr_Rp_plain = Rp_plain.to_ndarray()
    arr_Rp_z3 = Rp_z3.to_ndarray()

    print(f"\nRp shapes: plain={arr_Rp_plain.shape}, Z3={arr_Rp_z3.shape}")

    # SVD of Rp to see how much filtering is happening
    S_Rp_plain = np.linalg.svd(arr_Rp_plain, compute_uv=False)
    S_Rp_z3 = np.linalg.svd(arr_Rp_z3, compute_uv=False)

    print(f"\nRp singular values (1 = identity, <1 = filtering):")
    print(f"  Plain Rp: {S_Rp_plain}")
    print(f"  Z3 Rp:    {S_Rp_z3}")

    # Distance from identity
    I_plain = np.eye(arr_Rp_plain.shape[0])
    I_z3 = np.eye(arr_Rp_z3.shape[0])

    dist_plain = np.linalg.norm(arr_Rp_plain - I_plain)
    dist_z3 = np.linalg.norm(arr_Rp_z3 - I_z3)

    print(f"\n||Rp - I|| (how much GILT modifies the tensor):")
    print(f"  Plain: {dist_plain:.6f}")
    print(f"  Z3:    {dist_z3:.6f}")

    if dist_z3 > 2 * dist_plain:
        print(f"\n⚠️  Z3 GILT filter is {dist_z3/dist_plain:.1f}x more aggressive than plain!")
        print("  This could explain why Z3 diverges.")


def main():
    compare_gilt_step()
    compare_gilt_environment()
    compare_gilt_filter()

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
If the root cause is GILT:
- Environment singular values should be IDENTICAL (same physics)
- The Rp filter should be more aggressive for Z3
- After GILT step, Z3 tensor (converted to plain) should differ from plain

If the root cause is elsewhere:
- Environment singular values might differ
- Rp filters might be similar
- Issue might be in coarse-graining or truncation
""")


if __name__ == "__main__":
    main()
