#!/usr/bin/env python3
"""
Full implementation: GILT in spin basis for Z3 tensors.

This implements the fix for TensorZ3 + GILT-TNR instability:
1. Transform Z3 tensor to spin basis (inverse DFT)
2. Apply GILT filtering in spin basis
3. Transform back to charge basis (DFT)
4. Do coarse-graining in charge basis (for efficiency)

The key insight is that GILT's trace criterion works correctly in spin basis
but creates spurious modes in charge basis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D import gilttnr_step, get_envspec, optimize_Rp
from GiltTNR2D_Potts import get_initial_tensor_potts_relT
from tensors import Tensor, TensorZ3


def get_dft_matrix(N=3):
    """Get DFT matrix for Z_N."""
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega**(j*k) for k in range(N)] for j in range(N)]) / np.sqrt(N)
    return F


def z3_to_spin(T_z3):
    """
    Transform Z3 tensor from charge basis to spin basis.

    For TensorZ3, the indices are organized by charge sector.
    We need to apply DFT to each "charge block" separately.
    """
    arr = T_z3.to_ndarray()
    chi = arr.shape[0]

    if chi == 3:
        # Simple case: 3×3×3×3 tensor
        F = get_dft_matrix(3)
        Fdag = F.T.conj()
        result = np.einsum('ai,bj,ijkl,ck,dl->abcd', Fdag, Fdag, arr, F, F)
        return Tensor.from_ndarray(result)

    # For larger tensors with bond dimension chi:
    # The TensorZ3 has block structure where indices are grouped by charge.
    # If we have chi = n0 + n1 + n2 (dimensions per charge sector),
    # we need to apply DFT within each "3-tuple" of indices.

    # Check if we have TensorZ3 with known structure
    if hasattr(T_z3, 'shape') and hasattr(T_z3, 'qhape'):
        # Get the charge sector dimensions
        try:
            # For TensorZ3, shape is list of dicts or lists per leg
            # e.g., shape = [{0: d0, 1: d1, 2: d2}, ...]
            dims_per_sector = T_z3.shape[0]  # dimensions per charge sector for leg 0
            if isinstance(dims_per_sector, dict):
                n0 = dims_per_sector.get(0, 0)
                n1 = dims_per_sector.get(1, 0)
                n2 = dims_per_sector.get(2, 0)
            elif isinstance(dims_per_sector, (list, tuple)):
                n0, n1, n2 = dims_per_sector[0], dims_per_sector[1], dims_per_sector[2]
            else:
                # Assume equal distribution
                n0 = n1 = n2 = chi // 3

            # Build block-DFT matrix
            F = get_dft_matrix(3)
            Fdag = F.T.conj()

            # For each leg, construct the full transformation matrix
            # The transformation is: F_full = sum_q |spin_q><charge_q| ⊗ I_{n_q}
            # This is a bit complex; for now, use simpler approach:
            # Assume each charge sector maps to corresponding "spin component"

            # Build block-diagonal DFT: F_block[i,j] = F[q_i, q_j] * delta(offset_i, offset_j)
            # where q_i is the charge of index i

            # Create index-to-charge mapping
            charge_map = []
            offset_map = []
            for q, nq in enumerate([n0, n1, n2]):
                for k in range(nq):
                    charge_map.append(q)
                    offset_map.append(k)

            # Build transformation matrix
            F_full = np.zeros((chi, chi), dtype=complex)
            for i in range(chi):
                for j in range(chi):
                    if offset_map[i] == offset_map[j]:
                        F_full[i, j] = Fdag[charge_map[i], charge_map[j]]

            # Apply to tensor
            result = np.einsum('ai,bj,ijkl,ck,dl->abcd',
                               F_full, F_full, arr, F_full.conj(), F_full.conj())
            return Tensor.from_ndarray(result)

        except Exception as e:
            print(f"  z3_to_spin error: {e}")
            pass

    # Fallback: assume balanced sectors
    if chi % 3 == 0:
        n = chi // 3
        F = get_dft_matrix(3)
        Fdag = F.T.conj()

        # Block-DFT: F_block = F ⊗ I_n
        F_block = np.kron(Fdag, np.eye(n))

        result = np.einsum('ai,bj,ijkl,ck,dl->abcd',
                           F_block, F_block, arr, F_block.conj(), F_block.conj())
        return Tensor.from_ndarray(result)

    print(f"  Warning: Cannot transform chi={chi} tensor to spin basis")
    return None


def spin_to_z3(T_spin, original_z3=None):
    """
    Transform tensor from spin basis back to Z3 charge basis.

    If original_z3 is provided, we use its structure to create proper TensorZ3.
    """
    arr = T_spin.to_ndarray()
    chi = arr.shape[0]

    if chi == 3:
        F = get_dft_matrix(3)
        Fdag = F.T.conj()
        # Inverse of z3_to_spin: apply F on in-legs, F† on out-legs
        result = np.einsum('ai,bj,ijkl,ck,dl->abcd', F, F, arr, Fdag, Fdag)

        # Create TensorZ3
        dims = [1, 1, 1]
        qims = [0, 1, 2]
        try:
            return TensorZ3.from_ndarray(result, shape=[dims]*4, qhape=[qims]*4,
                                         dirs=[1, 1, -1, -1])
        except:
            return Tensor.from_ndarray(result)

    # For larger tensors
    if chi % 3 == 0:
        n = chi // 3
        F = get_dft_matrix(3)

        # Inverse block-DFT: F_block = F ⊗ I_n
        F_block = np.kron(F, np.eye(n))

        result = np.einsum('ai,bj,ijkl,ck,dl->abcd',
                           F_block, F_block, arr, F_block.conj(), F_block.conj())

        # Try to create TensorZ3
        dims = [n, n, n]
        qims = [0, 1, 2]
        try:
            return TensorZ3.from_ndarray(result, shape=[dims]*4, qhape=[qims]*4,
                                         dirs=[1, 1, -1, -1])
        except Exception as e:
            print(f"  spin_to_z3: Cannot create TensorZ3: {e}")
            return Tensor.from_ndarray(result)

    return Tensor.from_ndarray(arr)


def gilt_in_spin_basis(A_z3, pars):
    """
    Apply GILT to Z3 tensor, computing everything in spin basis.

    This is the key fix: GILT filtering happens in spin basis where
    trace structure is preserved correctly.
    """
    # Step 1: Transform to spin basis
    A_spin = z3_to_spin(A_z3)
    if A_spin is None:
        # Fallback for larger tensors - use standard GILT
        print("  Warning: Cannot transform to spin basis, using charge-basis GILT")
        return apply_standard_gilt(A_z3, pars)

    # Step 2: Create plain-tensor parameters
    pars_spin = dict(pars)
    pars_spin['symmetry_tensors'] = False

    # Step 3: Apply GILT in spin basis on all 4 edges
    # Following the structure of GiltTNR2D.apply_gilt()
    for where in ["S", "E", "N", "W"]:
        U, S = get_envspec(A_spin, A_spin, pars_spin, where=where)
        Rp, err = optimize_Rp(U, S, pars_spin)

        # Apply Rp to the appropriate leg
        if where == "S":
            A_spin = ncon([Rp, A_spin], [[-2, 1], [-1, 1, -3, -4]])
        elif where == "E":
            A_spin = ncon([Rp, A_spin], [[-3, 1], [-1, -2, 1, -4]])
        elif where == "N":
            A_spin = ncon([Rp, A_spin], [[-4, 1], [-1, -2, -3, 1]])
        elif where == "W":
            A_spin = ncon([Rp, A_spin], [[-1, 1], [1, -2, -3, -4]])

    # Step 4: Transform back to charge basis
    A_z3_filtered = spin_to_z3(A_spin)

    return A_z3_filtered


def apply_standard_gilt(A, pars):
    """Apply standard GILT in whatever basis the tensor is in."""
    for where in ["S", "E", "N", "W"]:
        U, S = get_envspec(A, A, pars, where=where)
        Rp, err = optimize_Rp(U, S, pars)

        if where == "S":
            A = ncon([Rp, A], [[-2, 1], [-1, 1, -3, -4]])
        elif where == "E":
            A = ncon([Rp, A], [[-3, 1], [-1, -2, 1, -4]])
        elif where == "N":
            A = ncon([Rp, A], [[-4, 1], [-1, -2, -3, 1]])
        elif where == "W":
            A = ncon([Rp, A], [[-1, 1], [1, -2, -3, -4]])

    return A


def gilttnr_step_spin_basis(A_z3, log_fact, pars):
    """
    One GILT-TNR step for Z3 tensor, with GILT done in spin basis.

    This combines:
    1. GILT filtering (in spin basis)
    2. Standard TRG coarse-graining (in charge basis)
    """
    # Apply GILT in spin basis
    A_filtered = gilt_in_spin_basis(A_z3, pars)

    # Now do standard TRG coarse-graining in charge basis
    # This is the split-contract-truncate step

    # For initial tensor (3x3x3x3), we need to do the full TRG step
    # Let's use standard gilttnr_step with gilt_eps=0 for just the TRG part
    pars_no_gilt = dict(pars)
    pars_no_gilt['gilt_eps'] = 0

    # If A_filtered is TensorZ3, use Z3 TRG; otherwise plain
    A_new, log_new = gilttnr_step(A_filtered, log_fact, pars_no_gilt)

    return A_new, log_new


def compare_approaches():
    """Compare standard Z3 GILT vs spin-basis GILT."""
    print("=" * 80)
    print("Full GILT-TNR: Spin Basis vs Charge Basis")
    print("=" * 80)

    chi = 16
    gilt_eps = 1e-6
    n_steps = 5

    pars_plain = {
        'q': 3, 'relT': 1.0, 'gilt_eps': gilt_eps,
        'cg_chis': [chi], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': False,
    }
    pars_z3 = dict(pars_plain)
    pars_z3['symmetry_tensors'] = True

    # Initial tensors
    A_plain = get_initial_tensor_potts_relT(pars_plain)
    A_z3_standard = get_initial_tensor_potts_relT(pars_z3)
    A_z3_spinbasis = get_initial_tensor_potts_relT(pars_z3)

    print(f"\nComparing {n_steps} GILT-TNR steps:")
    print(f"  - Plain tensor (reference)")
    print(f"  - Z3 standard (GILT in charge basis) - expected to fail")
    print(f"  - Z3 spin-basis (GILT in spin basis) - expected to work")

    print(f"\n{'Step':<6} {'Plain':<15} {'Z3 Standard':<15} {'Z3 SpinBasis':<15} {'Std/Plain':<12} {'Spin/Plain':<12}")
    print("-" * 85)

    log_plain = 0.0
    log_z3_std = 0.0
    log_z3_spin = 0.0

    for step in range(1, n_steps + 1):
        # Plain tensor - reference
        A_plain, log_plain = gilttnr_step(A_plain, log_plain, pars_plain)
        norm_plain = np.linalg.norm(A_plain.to_ndarray())

        # Z3 standard - GILT in charge basis (expected to fail)
        A_z3_standard, log_z3_std = gilttnr_step(A_z3_standard, log_z3_std, pars_z3)
        norm_z3_std = np.linalg.norm(A_z3_standard.to_ndarray())

        # Z3 spin-basis - GILT in spin basis
        try:
            A_z3_spinbasis, log_z3_spin = gilttnr_step_spin_basis(
                A_z3_spinbasis, log_z3_spin, pars_z3)
            norm_z3_spin = np.linalg.norm(A_z3_spinbasis.to_ndarray())
        except Exception as e:
            print(f"  Step {step} spin-basis error: {e}")
            norm_z3_spin = float('nan')

        ratio_std = norm_z3_std / norm_plain
        ratio_spin = norm_z3_spin / norm_plain if not np.isnan(norm_z3_spin) else float('nan')

        print(f"{step:<6} {norm_plain:<15.4e} {norm_z3_std:<15.4e} {norm_z3_spin:<15.4e} {ratio_std:<12.4f} {ratio_spin:<12.4f}")

    # Analyze final tensors
    print("\n" + "=" * 80)
    print("Final Tensor Analysis")
    print("=" * 80)

    # Compare singular value spectra
    def get_spectrum(A):
        arr = A.to_ndarray()
        d = arr.shape[0]
        mat = arr.reshape(d*d, d*d)
        return np.linalg.svd(mat, compute_uv=False)

    S_plain = get_spectrum(A_plain)
    S_z3_std = get_spectrum(A_z3_standard)
    if not np.isnan(norm_z3_spin):
        S_z3_spin = get_spectrum(A_z3_spinbasis)
    else:
        S_z3_spin = None

    print(f"\nTop 5 singular values (normalized):")
    print(f"{'Rank':<6} {'Plain':<15} {'Z3 Standard':<15} {'Z3 SpinBasis':<15}")
    print("-" * 55)
    for i in range(min(5, len(S_plain))):
        sp = S_plain[i] / S_plain[0]
        ss = S_z3_std[i] / S_z3_std[0] if len(S_z3_std) > i else 0
        sb = S_z3_spin[i] / S_z3_spin[0] if S_z3_spin is not None and len(S_z3_spin) > i else float('nan')
        print(f"{i:<6} {sp:<15.6f} {ss:<15.6f} {sb:<15.6f}")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
Expected behavior:
- Plain: Stable, serves as reference
- Z3 Standard: Diverges (norm ratio deviates from 1.0)
- Z3 SpinBasis: Should track plain tensor (norm ratio ~1.0)

If Z3 SpinBasis tracks plain, the fix works!
""")


if __name__ == "__main__":
    compare_approaches()
