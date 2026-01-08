#!/usr/bin/env python3
"""
GILT in spin basis via per-sector DFT transformation.

The approach:
1. For each charge sector q, get the block T_q of the tensor
2. Tensor T_q with DFT matrix: T_q ⊗ F (on each leg)
3. Take direct sum of all transformed blocks
4. Apply GILT in this combined spin basis
5. Transform back: apply F† to each sector
6. Reconstruct charge-basis tensor

Key insight: Each charge sector independently transforms to spin basis.
The direct sum combines them all for GILT computation.
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
    """DFT matrix for Z_N: F[j,k] = omega^(jk) / sqrt(N)."""
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega**(j*k) for k in range(N)] for j in range(N)]) / np.sqrt(N)
    return F


def extract_z3_blocks(T_z3):
    """
    Extract the blocks of a TensorZ3 by charge sector.

    For a TensorZ3 with 4 legs, each element T[i,j,k,l] is nonzero
    only if charge(i) + charge(j) - charge(k) - charge(l) ≡ 0 (mod 3).

    This function returns the blocks organized by (q0, q1, q2, q3).

    For the initial 3x3x3x3 tensor, this is straightforward.
    For larger tensors after TRG, we need to handle variable sector sizes.
    """
    if hasattr(T_z3, 'sects') and hasattr(T_z3, 'shape'):
        # This is a TensorZ3 with block structure
        # sects is a dict mapping (q0, q1, q2, q3) -> block array
        return dict(T_z3.sects)
    else:
        # Plain tensor or unknown structure
        return None


def tensor_with_dft(block, F):
    """
    Tensor a block with DFT matrix on each leg.

    For a 4-leg block T[i,j,k,l] of shape (n0, n1, n2, n3),
    the result has shape (n0*3, n1*3, n2*3, n3*3).

    T_expanded[i*3+s0, j*3+s1, k*3+s2, l*3+s3] = T[i,j,k,l] * F[s0,q0] * F[s1,q1] * F†[s2,q2] * F†[s3,q3]

    where q0,q1,q2,q3 are the charges of the block.

    Actually, for a block with fixed charges (q0,q1,q2,q3), we multiply by the DFT elements.
    """
    # For a block T_q with charge q on each leg:
    # In spin basis: sum over spins s, with coefficient F[s,q]

    # The tensor product structure:
    # T_spin = sum_{q} (F[:,q0] ⊗ F[:,q1] ⊗ F†[:,q2] ⊗ F†[:,q3]) T_q

    # This means each element of T_q gets multiplied by the appropriate DFT elements
    # and contributes to multiple spin-basis elements.
    pass  # We'll implement this differently below


def z3_to_spin_full(T_z3):
    """
    Transform TensorZ3 to plain tensor in spin basis.

    For initial tensor (chi=3, one element per charge combination):
    T_z3[q0, q1, q2, q3] is nonzero only for valid charge combinations.

    The spin basis tensor is:
    T_spin[s0, s1, s2, s3] = sum_{q0,q1,q2,q3} F†[s0,q0] F†[s1,q1] F[s2,q2] F[s3,q3] T_z3[q0,q1,q2,q3]

    For a 3x3x3x3 tensor, this is just:
    T_spin = einsum('ai,bj,ijkl,kc,ld->abcd', Fdag, Fdag, T_z3, F, F)
    """
    arr = T_z3.to_ndarray()

    if arr.shape == (3, 3, 3, 3):
        F = get_dft_matrix(3)
        Fdag = F.conj().T

        # Standard transformation: charge → spin
        # In-legs (0,1): apply F†
        # Out-legs (2,3): apply F
        result = np.einsum('ai,bj,ijkl,kc,ld->abcd', Fdag, Fdag, arr, F, F)
        return Tensor.from_ndarray(result)

    # For larger tensors with block structure:
    # We need to handle each charge sector and combine them

    # Check if this is a TensorZ3
    if not hasattr(T_z3, 'sects'):
        # Plain tensor - assume it's already in spin basis or can be treated as such
        return Tensor.from_ndarray(arr)

    # Get the shape information
    # T_z3.shape[leg] gives {q: dim_q} for each charge q in that leg
    leg_dims = []
    for leg in range(4):
        if isinstance(T_z3.shape[leg], dict):
            dims = T_z3.shape[leg]
        else:
            dims = {q: d for q, d in enumerate(T_z3.shape[leg])}
        leg_dims.append(dims)

    # Total spin-basis dimension for each leg
    # Each charge sector with dim n_q contributes n_q * 3 spin states
    # But actually, the spin states are shared... let me reconsider

    # In the charge basis, index i has a specific charge q_i.
    # In the spin basis, index s can have any spin value.
    # The transformation is: |s⟩ = sum_q F†[s,q] |q⟩

    # For a tensor with variable block sizes:
    # The total chi = sum_q n_q
    # Each charge index has chi/3 basis states per charge (approximately)

    # Build the transformation matrix U for each leg
    # U[s*n + offset, q_index] where q_index ranges over all charge indices

    F = get_dft_matrix(3)
    Fdag = F.conj().T

    # For now, use a simpler approach: if chi divisible by 3, use block DFT
    chi = arr.shape[0]
    if chi % 3 == 0:
        n = chi // 3
        # Assume equal sector sizes
        # Transform: U = F ⊗ I_n applied to charge index structure

        # Each leg has indices organized as [q=0 block, q=1 block, q=2 block]
        # The DFT mixes charge values: spin s = sum_q F†[s,q] * charge q

        # Build transformation matrix
        # For charge index c in sector q with offset k: c = q*n + k
        # For spin index s in component t with offset k: s = t*n + k
        # U[s, c] = F†[t, q] if offsets match, 0 otherwise

        U = np.zeros((chi, chi), dtype=complex)
        for q in range(3):
            for t in range(3):
                for k in range(n):
                    c_idx = q * n + k  # charge index
                    s_idx = t * n + k  # spin index
                    U[s_idx, c_idx] = Fdag[t, q]

        # Apply U to each leg
        result = np.einsum('ai,bj,ijkl,ck,dl->abcd', U, U, arr, U.conj(), U.conj())
        return Tensor.from_ndarray(result)

    # Fallback for non-divisible chi
    print(f"  Warning: chi={chi} not divisible by 3, cannot transform")
    return Tensor.from_ndarray(arr)


def spin_to_z3_full(T_spin, original_z3=None):
    """
    Transform spin-basis tensor back to charge basis TensorZ3.

    This is the inverse of z3_to_spin_full.
    """
    arr = T_spin.to_ndarray()
    chi = arr.shape[0]

    F = get_dft_matrix(3)
    Fdag = F.conj().T

    if chi == 3:
        # Simple case
        # Inverse: apply F on in-legs, F† on out-legs
        result = np.einsum('ai,bj,ijkl,ck,dl->abcd', F, F, arr, Fdag, Fdag)

        # Create TensorZ3
        try:
            return TensorZ3.from_ndarray(result, shape=[[1,1,1]]*4,
                                         qhape=[[0,1,2]]*4, dirs=[1,1,-1,-1])
        except:
            return Tensor.from_ndarray(result)

    if chi % 3 == 0:
        n = chi // 3
        # Build inverse transformation
        U = np.zeros((chi, chi), dtype=complex)
        for q in range(3):
            for t in range(3):
                for k in range(n):
                    c_idx = q * n + k
                    s_idx = t * n + k
                    U[s_idx, c_idx] = Fdag[t, q]

        # Inverse of U is U† (unitary)
        U_inv = U.conj().T

        result = np.einsum('ai,bj,ijkl,ck,dl->abcd', U_inv, U_inv, arr, U_inv.conj(), U_inv.conj())

        # Create TensorZ3
        try:
            return TensorZ3.from_ndarray(result, shape=[[n,n,n]]*4,
                                         qhape=[[0,1,2]]*4, dirs=[1,1,-1,-1])
        except Exception as e:
            print(f"  Cannot create TensorZ3: {e}")
            return Tensor.from_ndarray(result)

    return Tensor.from_ndarray(arr)


def gilt_in_spin(A_z3, pars):
    """Apply GILT to Z3 tensor in spin basis."""
    # Transform to spin basis
    A_spin = z3_to_spin_full(A_z3)

    # Plain tensor parameters
    pars_plain = dict(pars)
    pars_plain['symmetry_tensors'] = False

    # Apply GILT on all 4 edges
    for where in ["S", "E", "N", "W"]:
        U, S = get_envspec(A_spin, A_spin, pars_plain, where=where)
        Rp, _ = optimize_Rp(U, S, pars_plain)

        leg_map = {"W": 0, "S": 1, "E": 2, "N": 3}
        leg = leg_map[where]

        # Apply Rp to the appropriate leg
        if where == "S":
            A_spin = ncon([Rp, A_spin], [[-2, 1], [-1, 1, -3, -4]])
        elif where == "E":
            A_spin = ncon([Rp, A_spin], [[-3, 1], [-1, -2, 1, -4]])
        elif where == "N":
            A_spin = ncon([Rp, A_spin], [[-4, 1], [-1, -2, -3, 1]])
        elif where == "W":
            A_spin = ncon([Rp, A_spin], [[-1, 1], [1, -2, -3, -4]])

    # Transform back to charge basis
    A_z3_new = spin_to_z3_full(A_spin, A_z3)

    return A_z3_new


def gilttnr_step_spin(A_z3, log_fact, pars):
    """One GILT-TNR step with GILT in spin basis."""
    # GILT in spin basis
    A_filtered = gilt_in_spin(A_z3, pars)

    # TRG coarse-graining (no GILT)
    pars_no_gilt = dict(pars)
    pars_no_gilt['gilt_eps'] = 0

    A_new, log_new = gilttnr_step(A_filtered, log_fact, pars_no_gilt)

    return A_new, log_new


def test_transform():
    """Test the DFT transform on initial tensor."""
    print("=" * 80)
    print("Testing DFT Transform")
    print("=" * 80)

    pars_plain = {
        'q': 3, 'relT': 1.0, 'gilt_eps': 1e-6,
        'cg_chis': [16], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': False,
    }
    pars_z3 = dict(pars_plain)
    pars_z3['symmetry_tensors'] = True

    A_plain = get_initial_tensor_potts_relT(pars_plain)
    A_z3 = get_initial_tensor_potts_relT(pars_z3)

    # Transform Z3 to spin
    A_z3_spin = z3_to_spin_full(A_z3)

    arr_plain = A_plain.to_ndarray()
    arr_z3_spin = A_z3_spin.to_ndarray()

    diff = np.linalg.norm(arr_plain - arr_z3_spin) / np.linalg.norm(arr_plain)
    print(f"\n||plain - Z3→spin|| / ||plain|| = {diff:.2e}")

    if diff < 1e-10:
        print("✓ Transform correct!")
    else:
        print("✗ Transform has error")

    # Test round-trip
    A_z3_back = spin_to_z3_full(A_z3_spin)
    arr_z3_back = A_z3_back.to_ndarray()
    arr_z3_orig = A_z3.to_ndarray()

    diff_rt = np.linalg.norm(arr_z3_orig - arr_z3_back) / np.linalg.norm(arr_z3_orig)
    print(f"Round-trip error: {diff_rt:.2e}")

    # Compare GILT traces
    print("\n--- GILT Environment Comparison ---")
    U_plain, _ = get_envspec(A_plain, A_plain, pars_plain, where="S")
    U_z3_spin, _ = get_envspec(A_z3_spin, A_z3_spin, pars_plain, where="S")

    t_plain = ncon(U_plain, [1, 1, -1]).to_ndarray()
    t_z3_spin = ncon(U_z3_spin, [1, 1, -1]).to_ndarray()

    print(f"Modes with |t|>0.1: plain={np.sum(np.abs(t_plain)>0.1)}, Z3→spin={np.sum(np.abs(t_z3_spin)>0.1)}")


def compare_flow():
    """Compare GILT-TNR flow."""
    print("\n" + "=" * 80)
    print("GILT-TNR Flow Comparison")
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

    A_plain = get_initial_tensor_potts_relT(pars_plain)
    A_z3_std = get_initial_tensor_potts_relT(pars_z3)
    A_z3_spin = get_initial_tensor_potts_relT(pars_z3)

    log_p, log_std, log_spin = 0.0, 0.0, 0.0

    print(f"\n{'Step':<6} {'Plain':<15} {'Z3 Std':<15} {'Z3 Spin':<15} {'Std/P':<10} {'Spin/P':<10}")
    print("-" * 80)

    for step in range(1, n_steps + 1):
        A_plain, log_p = gilttnr_step(A_plain, log_p, pars_plain)
        norm_p = np.linalg.norm(A_plain.to_ndarray())

        A_z3_std, log_std = gilttnr_step(A_z3_std, log_std, pars_z3)
        norm_std = np.linalg.norm(A_z3_std.to_ndarray())

        try:
            A_z3_spin, log_spin = gilttnr_step_spin(A_z3_spin, log_spin, pars_z3)
            norm_spin = np.linalg.norm(A_z3_spin.to_ndarray())
        except Exception as e:
            print(f"  Step {step} spin error: {e}")
            norm_spin = float('nan')

        r_std = norm_std / norm_p
        r_spin = norm_spin / norm_p if not np.isnan(norm_spin) else float('nan')

        print(f"{step:<6} {norm_p:<15.4e} {norm_std:<15.4e} {norm_spin:<15.4e} {r_std:<10.4f} {r_spin:<10.4f}")


if __name__ == "__main__":
    test_transform()
    compare_flow()
