#!/usr/bin/env python3
"""
Fix GILT for Z3 equivariant tensors by applying GILT in spin basis.

CORRECT APPROACH (per-sector tensor product):
1. For each charge sector q with dimension n_q:
   - Tensor the sector block with DFT vector F[:,q] → 3×n_q matrix
   - This maps: |q,α⟩ → Σ_s F[s,q] |s,α⟩
2. Take direct sum across all sectors: Σ_q (3 × n_q) = 3 × Σ_q n_q = 3χ
   Wait, that grows dimension! Actually...

REVISED APPROACH:
The spin basis and charge basis have the SAME dimension χ.
- Charge basis: index i ∈ {0,...,χ-1}, with charge(i) determined by which sector
- Spin basis: index i ∈ {0,...,χ-1}, representing mixed spin states

For a leg with sector dims [n_0, n_1, n_2] where χ = n_0 + n_1 + n_2:
- Charge index c in sector q has offset k: c = Σ_{j<q} n_j + k
- The transformation matrix U[s_idx, c_idx] mixes charge contributions

The key insight: Within each sector, the "multiplicity" indices (α) don't mix.
Only the charge values mix via DFT. So:
  U = F† ⊗ I (block-wise on each multiplicity layer)

For UNBALANCED sectors:
- We need to handle the different sector sizes carefully
- Only the first min(n_0, n_1, n_2) multiplicities transform cleanly
- Extra multiplicities in larger sectors get mapped separately
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


def get_sector_dims(T_z3, leg=0):
    """
    Get sector dimensions for a TensorZ3 leg.

    Returns dict {charge: dimension} for the given leg.
    """
    if hasattr(T_z3, 'shape') and isinstance(T_z3.shape[leg], dict):
        return dict(T_z3.shape[leg])
    else:
        # Assume equal sectors
        chi = T_z3.to_ndarray().shape[leg]
        n = chi // 3
        return {0: n, 1: n, 2: chi - 2*n}


def build_transform_matrix(sector_dims, N=3):
    """
    Build the transformation matrix from charge basis to spin basis.

    For sector dimensions {0: n0, 1: n1, 2: n2}:
    - Total chi = n0 + n1 + n2
    - Charge index c = sum_{j<q} n_j + k  (where k < n_q)
    - The transformation uses F† on the charge values

    The transformation is:
    |spin_s, multiplicity_m⟩ = sum_q F†[s,q] |charge_q, multiplicity_m⟩

    For unequal sectors, we need a block-wise transformation.
    """
    F = get_dft_matrix(N)
    Fdag = F.conj().T

    chi = sum(sector_dims.values())
    ns = [sector_dims.get(q, 0) for q in range(N)]

    # Build the transformation matrix
    # Each charge sector q has ns[q] basis states
    # In charge basis: index c = sum_{j<q} ns[j] + k for charge q, offset k
    # In spin basis: we want to mix charge values

    # For balanced sectors (all ns equal):
    # U = F† ⊗ I_n
    # U[s*n + k, q*n + k] = F†[s, q]

    # For unbalanced sectors, we use a generalized approach:
    # The key insight is that the DFT only mixes charge values, not multiplicities
    # So we can treat each "layer" of multiplicities separately

    # Build offsets for charge index → (charge, offset)
    charge_start = {}
    offset = 0
    for q in range(N):
        charge_start[q] = offset
        offset += ns[q]

    # The transformation matrix U has shape (chi, chi)
    # but we need to be careful about what it means

    # Actually, for unbalanced sectors, the transformation is more subtle.
    # Let's use a different approach: embed into balanced space, transform, project back

    # Find minimum sector size
    n_min = min(ns) if all(n > 0 for n in ns) else 0
    n_max = max(ns)

    if n_min == 0:
        # Some sector is empty - return identity
        return np.eye(chi, dtype=complex)

    # Strategy: The DFT transformation mixes charge values.
    # For multiplicity index k (within a sector), the transformation is:
    # |spin_s, k⟩ = sum_q F†[s,q] |charge_q, k⟩
    #
    # But this only works if all sectors have at least n_min states.
    # Extra states in larger sectors need special handling.

    U = np.zeros((chi, chi), dtype=complex)

    # For the first n_min multiplicities in each sector, do proper DFT
    for k in range(n_min):
        for s in range(N):
            for q in range(N):
                spin_idx = s * n_min + k  # Spin index (first n_min*3 indices)
                charge_idx = charge_start[q] + k  # Charge index
                U[spin_idx, charge_idx] = Fdag[s, q]

    # Handle extra multiplicities in larger sectors
    # These don't mix with other charges, so map them to identity
    extra_idx = N * n_min
    for q in range(N):
        for k in range(n_min, ns[q]):
            charge_idx = charge_start[q] + k
            if extra_idx < chi:
                U[extra_idx, charge_idx] = 1.0
                extra_idx += 1

    return U


def transform_to_spin(T_z3):
    """
    Transform TensorZ3 to spin basis (plain tensor).

    This handles unbalanced sector dimensions properly.
    """
    arr = T_z3.to_ndarray()

    # Get sector dimensions for each leg
    dims = [get_sector_dims(T_z3, leg) for leg in range(4)]

    # Build transformation matrices for each leg
    # Legs 0,1 are "in" legs (covariant), legs 2,3 are "out" legs (contravariant)
    U0 = build_transform_matrix(dims[0])
    U1 = build_transform_matrix(dims[1])
    U2 = build_transform_matrix(dims[2])
    U3 = build_transform_matrix(dims[3])

    # Apply transformation
    # T_spin = U0 ⊗ U1 ⊗ T_charge ⊗ U2† ⊗ U3†
    result = np.einsum('ai,bj,ijkl,ck,dl->abcd',
                       U0, U1, arr, U2.conj(), U3.conj())

    return Tensor.from_ndarray(result), (U0, U1, U2, U3)


def transform_to_charge(T_spin, transforms, original_z3):
    """
    Transform spin-basis tensor back to charge basis TensorZ3.
    """
    arr = T_spin.to_ndarray()
    U0, U1, U2, U3 = transforms

    # Inverse transform: U† ⊗ U† ⊗ T_spin ⊗ U ⊗ U
    # (U is unitary, so U† = U^{-1})
    U0_inv = U0.conj().T
    U1_inv = U1.conj().T
    U2_inv = U2.conj().T
    U3_inv = U3.conj().T

    result = np.einsum('ai,bj,ijkl,ck,dl->abcd',
                       U0_inv, U1_inv, arr, U2_inv.conj(), U3_inv.conj())

    # Reconstruct TensorZ3
    # Get the block structure from original tensor
    if hasattr(original_z3, 'shape') and hasattr(original_z3, 'qhape'):
        try:
            shape = [[d for d in leg.values()] if isinstance(leg, dict) else leg
                     for leg in original_z3.shape]
            qhape = [[q for q in leg.values()] if isinstance(leg, dict) else leg
                     for leg in original_z3.qhape]
            dirs = original_z3.dirs if hasattr(original_z3, 'dirs') else [1, 1, -1, -1]

            return TensorZ3.from_ndarray(result.real, shape=shape, qhape=qhape, dirs=dirs)
        except Exception as e:
            print(f"  Warning: Cannot reconstruct TensorZ3: {e}")

    return Tensor.from_ndarray(result)


def gilt_in_spin_basis(A_z3, pars):
    """
    Apply GILT to Z3 tensor by transforming to spin basis.

    This is the key fix: GILT uses trace-based mode identification,
    which works correctly in spin basis but not in charge basis.
    """
    # Transform to spin basis
    A_spin, transforms = transform_to_spin(A_z3)

    # Plain tensor parameters for GILT
    pars_plain = dict(pars)
    pars_plain['symmetry_tensors'] = False

    # Apply GILT on all 4 edges
    for where in ["S", "E", "N", "W"]:
        U, S = get_envspec(A_spin, A_spin, pars_plain, where=where)
        Rp, _ = optimize_Rp(U, S, pars_plain)

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
    A_z3_new = transform_to_charge(A_spin, transforms, A_z3)

    return A_z3_new


def gilttnr_step_fixed(A_z3, log_fact, pars):
    """
    One GILT-TNR step with GILT applied in spin basis.

    Flow:
    1. Transform Z3 to spin basis
    2. Apply GILT (in spin basis)
    3. Transform back to Z3
    4. Apply TRG coarse-graining (with Z3 tensors)
    """
    # GILT in spin basis
    A_filtered = gilt_in_spin_basis(A_z3, pars)

    # TRG coarse-graining (no additional GILT)
    pars_no_gilt = dict(pars)
    pars_no_gilt['gilt_eps'] = 0

    A_new, log_new = gilttnr_step(A_filtered, log_fact, pars_no_gilt)

    return A_new, log_new


def test_initial_transform():
    """Test the transformation on initial tensor."""
    print("=" * 80)
    print("Testing Initial Tensor Transformation")
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
    A_spin, transforms = transform_to_spin(A_z3)

    arr_plain = A_plain.to_ndarray()
    arr_spin = A_spin.to_ndarray()

    diff = np.linalg.norm(arr_plain - arr_spin) / np.linalg.norm(arr_plain)
    print(f"\n||plain - Z3→spin|| / ||plain|| = {diff:.2e}")

    if diff < 1e-10:
        print("✓ Transform correct!")
    else:
        print("✗ Transform has error")

    # Test round-trip
    A_z3_back = transform_to_charge(A_spin, transforms, A_z3)
    arr_z3_back = A_z3_back.to_ndarray()
    arr_z3_orig = A_z3.to_ndarray()

    diff_rt = np.linalg.norm(arr_z3_orig - arr_z3_back) / np.linalg.norm(arr_z3_orig)
    print(f"Round-trip error: {diff_rt:.2e}")


def compare_gilt_modes():
    """Compare GILT modes between charge basis and spin basis."""
    print("\n" + "=" * 80)
    print("GILT Mode Comparison: Charge vs Spin Basis")
    print("=" * 80)

    pars = {
        'q': 3, 'relT': 1.0, 'gilt_eps': 1e-6,
        'cg_chis': [16], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': True,
    }

    A_z3 = get_initial_tensor_potts_relT(pars)
    A_spin, _ = transform_to_spin(A_z3)

    pars_plain = dict(pars)
    pars_plain['symmetry_tensors'] = False

    print("\n--- Initial Tensor ---")
    print(f"{'Where':<6} {'Modes (charge)':<16} {'Modes (spin)':<16} {'Ratio':<10}")
    print("-" * 50)

    for where in ["S", "E", "N", "W"]:
        # Charge basis modes (using Z3 env computation)
        U_charge, _ = get_envspec(A_z3, A_z3, pars, where=where)
        t_charge = ncon(U_charge, [1, 1, -1]).to_ndarray()
        n_charge = np.sum(np.abs(t_charge) > 0.1)

        # Spin basis modes (using plain env computation)
        U_spin, _ = get_envspec(A_spin, A_spin, pars_plain, where=where)
        t_spin = ncon(U_spin, [1, 1, -1]).to_ndarray()
        n_spin = np.sum(np.abs(t_spin) > 0.1)

        ratio = n_charge / n_spin if n_spin > 0 else float('inf')
        print(f"{where:<6} {n_charge:<16} {n_spin:<16} {ratio:<10.2f}")


def compare_flows():
    """Compare GILT-TNR flows: standard Z3 vs fixed Z3."""
    print("\n" + "=" * 80)
    print("GILT-TNR Flow Comparison: Standard Z3 vs Fixed Z3")
    print("=" * 80)

    chi = 16
    gilt_eps = 1e-6
    n_steps = 6

    pars_plain = {
        'q': 3, 'relT': 1.0, 'gilt_eps': gilt_eps,
        'cg_chis': [chi], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': False,
    }
    pars_z3 = dict(pars_plain)
    pars_z3['symmetry_tensors'] = True

    A_plain = get_initial_tensor_potts_relT(pars_plain)
    A_z3_std = get_initial_tensor_potts_relT(pars_z3)
    A_z3_fix = get_initial_tensor_potts_relT(pars_z3)

    log_p, log_std, log_fix = 0.0, 0.0, 0.0

    print(f"\n{'Step':<6} {'||A_plain||':<14} {'||A_z3_std||':<14} {'||A_z3_fix||':<14} "
          f"{'Std/P':<10} {'Fix/P':<10}")
    print("-" * 80)

    for step in range(1, n_steps + 1):
        # Plain tensor flow
        A_plain, log_p = gilttnr_step(A_plain, log_p, pars_plain)
        norm_p = np.linalg.norm(A_plain.to_ndarray())

        # Standard Z3 flow (has GILT mode issue)
        try:
            A_z3_std, log_std = gilttnr_step(A_z3_std, log_std, pars_z3)
            norm_std = np.linalg.norm(A_z3_std.to_ndarray())
        except Exception as e:
            print(f"  Std error at step {step}: {e}")
            norm_std = float('nan')

        # Fixed Z3 flow (GILT in spin basis)
        try:
            A_z3_fix, log_fix = gilttnr_step_fixed(A_z3_fix, log_fix, pars_z3)
            norm_fix = np.linalg.norm(A_z3_fix.to_ndarray())
        except Exception as e:
            print(f"  Fix error at step {step}: {e}")
            norm_fix = float('nan')

        r_std = norm_std / norm_p if not np.isnan(norm_std) else float('nan')
        r_fix = norm_fix / norm_p if not np.isnan(norm_fix) else float('nan')

        print(f"{step:<6} {norm_p:<14.4e} {norm_std:<14.4e} {norm_fix:<14.4e} "
              f"{r_std:<10.4f} {r_fix:<10.4f}")


if __name__ == "__main__":
    test_initial_transform()
    compare_gilt_modes()
    compare_flows()
