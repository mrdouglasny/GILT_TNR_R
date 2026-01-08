#!/usr/bin/env python3
"""
GILT in spin basis for Z3 tensors - Version 2.

The key insight is that TensorZ3.to_ndarray() organizes indices by charge sector:
- Indices [0, n0) have charge 0
- Indices [n0, n0+n1) have charge 1
- Indices [n0+n1, n0+n1+n2) have charge 2

To transform to spin basis:
1. Build a transformation matrix U where U[i,j] = F†[q_i, q_j] if they're in the same "slot"
2. Apply U to each leg of the tensor
3. This gives a plain tensor in spin basis

The transformation is essentially: for each "multiplicity index" m, apply DFT across charges.
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


def get_sector_dims(T_z3, leg=0):
    """
    Get dimensions of each charge sector for a leg of TensorZ3.

    Returns: (n0, n1, n2) - dimensions of charges 0, 1, 2
    """
    if not hasattr(T_z3, 'shape'):
        # Plain tensor
        chi = T_z3.to_ndarray().shape[leg]
        if chi == 3:
            return (1, 1, 1)
        elif chi % 3 == 0:
            n = chi // 3
            return (n, n, n)
        else:
            return None

    # TensorZ3 has shape attribute that gives sector dimensions
    # shape is a list (one per leg) of lists/dicts
    try:
        leg_shape = T_z3.shape[leg]
        if isinstance(leg_shape, dict):
            return (leg_shape.get(0, 0), leg_shape.get(1, 0), leg_shape.get(2, 0))
        elif isinstance(leg_shape, (list, tuple)):
            return tuple(leg_shape)
        else:
            # Try to infer from total dimension
            chi = T_z3.to_ndarray().shape[leg]
            return (chi // 3, chi // 3, chi - 2*(chi//3))
    except:
        chi = T_z3.to_ndarray().shape[leg]
        return (chi // 3, chi // 3, chi - 2*(chi//3))


def build_dft_transform(sector_dims):
    """
    Build the DFT transformation matrix for a leg with given sector dimensions.

    For sector_dims = (n0, n1, n2), the total dimension is chi = n0 + n1 + n2.
    The transformation matrix U has shape (chi, chi).

    The transformation maps: charge basis → spin basis
    U[spin_idx, charge_idx] = F†[spin_charge, charge] * delta(spin_offset, charge_offset)

    where spin_idx = spin_charge * n_max + spin_offset (approximately)
    """
    n0, n1, n2 = sector_dims
    chi = n0 + n1 + n2

    F = get_dft_matrix(3)
    Fdag = F.conj().T

    # For unequal sector sizes, we need to handle this carefully.
    # The idea: each index i in charge basis has (charge q_i, offset within sector)
    # We want to map to spin basis where spin s has n_s copies

    # For simplicity, assume sectors have similar sizes and use the minimum
    n_min = min(n0, n1, n2)

    if n_min == 0:
        # Some sector is empty - can't do proper DFT
        return None

    # Build transformation for the common part
    # U transforms (q, offset) → (s, offset) via F†[s, q]

    # Create index mappings
    # In charge basis: i → (q_i, offset_i)
    charge_starts = [0, n0, n0 + n1]

    # Build U matrix
    U = np.zeros((chi, chi), dtype=complex)

    for q in range(3):  # For each charge sector
        for offset in range(min(sector_dims[q], n_min)):
            i = charge_starts[q] + offset  # charge basis index

            for s in range(3):  # For each spin value
                # In spin basis, "spin s, offset" maps to index...
                # We'll use a similar layout: spin_starts[s] + offset
                spin_starts = [0, n_min, 2*n_min]
                j = spin_starts[s] + offset

                if j < chi:
                    U[j, i] = Fdag[s, q]

    return U


def z3_to_spin_v2(T_z3):
    """
    Transform Z3 tensor from charge basis to spin basis.

    For each leg, applies DFT transformation.
    """
    arr = T_z3.to_ndarray()

    # Get sector dimensions for each leg
    sector_dims = []
    for leg in range(4):
        dims = get_sector_dims(T_z3, leg)
        if dims is None:
            return None
        sector_dims.append(dims)

    # Build transformation matrices for each leg
    Us = []
    for leg in range(4):
        U = build_dft_transform(sector_dims[leg])
        if U is None:
            return None
        Us.append(U)

    # Apply transformations: T_spin = U0 U1 T_charge U2† U3†
    # (U on in-legs, U† on out-legs)
    result = np.einsum('ai,bj,ijkl,ck,dl->abcd',
                       Us[0], Us[1], arr, Us[2].conj(), Us[3].conj())

    return Tensor.from_ndarray(result)


def spin_to_z3_v2(T_spin, sector_dims_list):
    """
    Transform tensor from spin basis back to Z3 charge basis.

    sector_dims_list: list of (n0, n1, n2) for each of 4 legs
    """
    arr = T_spin.to_ndarray()

    # Build inverse transformation matrices
    Us_inv = []
    for leg in range(4):
        U = build_dft_transform(sector_dims_list[leg])
        if U is None:
            return None
        # Inverse of U is U† (unitary)
        Us_inv.append(U.conj().T)

    # Apply inverse transformations
    result = np.einsum('ai,bj,ijkl,ck,dl->abcd',
                       Us_inv[0], Us_inv[1], arr, Us_inv[2].conj(), Us_inv[3].conj())

    # Create TensorZ3
    dims_per_leg = [list(sd) for sd in sector_dims_list]
    qims = [0, 1, 2]
    try:
        return TensorZ3.from_ndarray(result, shape=dims_per_leg, qhape=[qims]*4,
                                     dirs=[1, 1, -1, -1])
    except Exception as e:
        print(f"  Warning: Cannot create TensorZ3: {e}")
        return Tensor.from_ndarray(result)


def gilt_step_spin_basis(A_z3, pars):
    """
    Apply GILT to Z3 tensor in spin basis.

    1. Get sector dimensions
    2. Transform to spin basis
    3. Apply GILT
    4. Transform back
    """
    # Get sector dimensions
    sector_dims_list = [get_sector_dims(A_z3, leg) for leg in range(4)]

    # Transform to spin basis
    A_spin = z3_to_spin_v2(A_z3)
    if A_spin is None:
        print("  Warning: Cannot transform to spin basis")
        return A_z3

    # Parameters for plain tensor GILT
    pars_plain = dict(pars)
    pars_plain['symmetry_tensors'] = False

    # Apply GILT on all edges in spin basis
    for where in ["S", "E", "N", "W"]:
        U, S = get_envspec(A_spin, A_spin, pars_plain, where=where)
        Rp, err = optimize_Rp(U, S, pars_plain)

        if where == "S":
            A_spin = ncon([Rp, A_spin], [[-2, 1], [-1, 1, -3, -4]])
        elif where == "E":
            A_spin = ncon([Rp, A_spin], [[-3, 1], [-1, -2, 1, -4]])
        elif where == "N":
            A_spin = ncon([Rp, A_spin], [[-4, 1], [-1, -2, -3, 1]])
        elif where == "W":
            A_spin = ncon([Rp, A_spin], [[-1, 1], [1, -2, -3, -4]])

    # Transform back to charge basis
    A_z3_new = spin_to_z3_v2(A_spin, sector_dims_list)

    return A_z3_new


def gilttnr_step_spin(A_z3, log_fact, pars):
    """
    One GILT-TNR step with GILT done in spin basis.
    """
    # Apply GILT in spin basis
    A_filtered = gilt_step_spin_basis(A_z3, pars)

    # Do TRG coarse-graining (with gilt_eps=0 to skip GILT)
    pars_no_gilt = dict(pars)
    pars_no_gilt['gilt_eps'] = 0

    A_new, log_new = gilttnr_step(A_filtered, log_fact, pars_no_gilt)

    return A_new, log_new


def test_initial_transform():
    """Test that DFT transform works on initial tensor."""
    print("=" * 80)
    print("Testing DFT Transform on Initial Tensor")
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
    A_z3_spin = z3_to_spin_v2(A_z3)

    if A_z3_spin is not None:
        arr_plain = A_plain.to_ndarray()
        arr_z3_spin = A_z3_spin.to_ndarray()

        diff = np.linalg.norm(arr_plain - arr_z3_spin) / np.linalg.norm(arr_plain)
        print(f"\n||plain - Z3→spin|| / ||plain|| = {diff:.2e}")

        if diff < 1e-10:
            print("✓ Transform is correct!")
        else:
            print("✗ Transform has error")
    else:
        print("✗ Transform failed")

    # Compare GILT environments
    print("\n--- GILT Environment Comparison ---")
    U_plain, S_plain = get_envspec(A_plain, A_plain, pars_plain, where="S")
    if A_z3_spin is not None:
        U_z3_spin, S_z3_spin = get_envspec(A_z3_spin, A_z3_spin, pars_plain, where="S")

        t_plain = ncon(U_plain, [1, 1, -1]).to_ndarray()
        t_z3_spin = ncon(U_z3_spin, [1, 1, -1]).to_ndarray()

        n_plain = np.sum(np.abs(t_plain) > 0.1)
        n_z3_spin = np.sum(np.abs(t_z3_spin) > 0.1)
        print(f"Modes with |t|>0.1: plain={n_plain}, Z3→spin={n_z3_spin}")


def compare_full_flow():
    """Compare full GILT-TNR flow."""
    print("\n" + "=" * 80)
    print("Full GILT-TNR Comparison")
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

    log_plain = 0.0
    log_z3_std = 0.0
    log_z3_spin = 0.0

    print(f"\n{'Step':<6} {'Plain':<15} {'Z3 Std':<15} {'Z3 Spin':<15} {'Std/P':<10} {'Spin/P':<10}")
    print("-" * 80)

    for step in range(1, n_steps + 1):
        # Plain reference
        A_plain, log_plain = gilttnr_step(A_plain, log_plain, pars_plain)
        norm_plain = np.linalg.norm(A_plain.to_ndarray())

        # Z3 standard
        A_z3_std, log_z3_std = gilttnr_step(A_z3_std, log_z3_std, pars_z3)
        norm_z3_std = np.linalg.norm(A_z3_std.to_ndarray())

        # Z3 with spin-basis GILT
        try:
            A_z3_spin, log_z3_spin = gilttnr_step_spin(A_z3_spin, log_z3_spin, pars_z3)
            norm_z3_spin = np.linalg.norm(A_z3_spin.to_ndarray())
        except Exception as e:
            print(f"  Step {step} error: {e}")
            norm_z3_spin = float('nan')

        ratio_std = norm_z3_std / norm_plain
        ratio_spin = norm_z3_spin / norm_plain if not np.isnan(norm_z3_spin) else float('nan')

        print(f"{step:<6} {norm_plain:<15.4e} {norm_z3_std:<15.4e} {norm_z3_spin:<15.4e} {ratio_std:<10.4f} {ratio_spin:<10.4f}")


if __name__ == "__main__":
    test_initial_transform()
    compare_full_flow()
