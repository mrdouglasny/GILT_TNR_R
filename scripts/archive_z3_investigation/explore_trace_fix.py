#!/usr/bin/env python3
"""
Explore fixing the trace computation for block-diagonal tensors.

The issue: GILT uses trace(U_i) to identify CDL modes.
For block-diagonal U, the trace only sees diagonal blocks.

Key insight from user: charges q and -q are conjugates in Z_N.
For Z_3: q=1 and q=2 are conjugates (2 ≡ -1 mod 3).

This means:
- The (q=0) block is "real" (self-conjugate)
- The (q=1) and (q=2) blocks are conjugate pairs

The trace in spin basis captures contributions from ALL spin values.
The trace in charge basis only captures contributions within each charge sector.

Potential fix: Modify the trace computation to account for this.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D import get_envspec, optimize_Rp
from GiltTNR2D_Potts import get_initial_tensor_potts_relT
from tensors import Tensor, TensorZ3


def get_dft_matrix(N=3):
    """DFT matrix for Z_N."""
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega**(j*k) for k in range(N)] for j in range(N)]) / np.sqrt(N)
    return F


def analyze_u_matrix_structure(U, name=""):
    """Analyze the structure of U matrix from environment spectrum."""
    arr = U.to_ndarray()
    chi = arr.shape[0]
    n_modes = arr.shape[2]

    print(f"\n--- {name} U Matrix Analysis ---")
    print(f"Shape: {arr.shape} (chi={chi}, n_modes={n_modes})")

    # For each mode, analyze the chi×chi matrix
    for i in range(min(5, n_modes)):
        U_i = arr[:, :, i]

        # Trace
        tr = np.trace(U_i)

        # Check if it's diagonal
        off_diag = np.linalg.norm(U_i - np.diag(np.diag(U_i)))

        # Check block structure (for chi=3, blocks are 1×1)
        if chi == 3:
            diag_vals = np.diag(U_i)
            print(f"  Mode {i}: trace={tr:.4f}, |t|={abs(tr):.4f}, diag={diag_vals}")
        else:
            print(f"  Mode {i}: trace={tr:.4f}, |t|={abs(tr):.4f}, off_diag_norm={off_diag:.4f}")


def compute_corrected_trace(U_z3):
    """
    Compute trace that accounts for Z3 block structure.

    For Z3, the U matrix in charge basis is block-diagonal.
    The trace in spin basis would be different.

    If U_charge = diag(U_0, U_1, U_2) in block form, then
    U_spin = F† U_charge F

    Trace(U_spin) = Trace(F† U_charge F) = Trace(U_charge F F†) = Trace(U_charge)

    Wait - trace is invariant under similarity transform!
    So Trace(U_spin) = Trace(U_charge).

    The issue must be in HOW the modes are organized, not the trace values.
    """
    arr = U_z3.to_ndarray()
    chi = arr.shape[0]
    n_modes = arr.shape[2]

    traces = []
    for i in range(n_modes):
        U_i = arr[:, :, i]
        tr = np.trace(U_i)
        traces.append(tr)

    return np.array(traces)


def compare_mode_structure():
    """Compare the mode structure between plain and Z3."""
    print("=" * 80)
    print("Mode Structure Comparison: Plain vs Z3")
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

    # Get environments
    U_plain, S_plain = get_envspec(A_plain, A_plain, pars_plain, where="S")
    U_z3, S_z3 = get_envspec(A_z3, A_z3, pars_z3, where="S")

    analyze_u_matrix_structure(U_plain, "Plain")
    analyze_u_matrix_structure(U_z3, "Z3")

    # Compare the actual U matrices
    arr_U_plain = U_plain.to_ndarray()
    arr_U_z3 = U_z3.to_ndarray()

    print("\n--- Direct U Matrix Comparison ---")
    print(f"Plain U shape: {arr_U_plain.shape}")
    print(f"Z3 U shape: {arr_U_z3.shape}")

    # For initial tensor (chi=3), compare mode by mode
    if arr_U_plain.shape == arr_U_z3.shape:
        chi = arr_U_plain.shape[0]
        n_modes = arr_U_plain.shape[2]

        # Apply DFT to Z3 U matrices to transform to spin basis
        F = get_dft_matrix(3)
        Fdag = F.conj().T

        print(f"\nComparing first 5 modes (chi={chi}):")
        print(f"{'Mode':<6} {'|t|_plain':<12} {'|t|_z3':<12} {'|t|_z3→spin':<12} {'U_match':<12}")
        print("-" * 60)

        for i in range(min(5, n_modes)):
            U_plain_i = arr_U_plain[:, :, i]
            U_z3_i = arr_U_z3[:, :, i]

            # Transform Z3 to spin basis
            # U_spin = F† U_charge F (similarity transform)
            U_z3_spin_i = Fdag @ U_z3_i @ F

            t_plain = np.trace(U_plain_i)
            t_z3 = np.trace(U_z3_i)
            t_z3_spin = np.trace(U_z3_spin_i)  # Should equal t_z3 (trace invariant)

            # Compare U matrices (up to phase/permutation)
            # Frobenius norm of difference
            diff = np.linalg.norm(U_plain_i - U_z3_spin_i) / np.linalg.norm(U_plain_i)

            print(f"{i:<6} {abs(t_plain):<12.4f} {abs(t_z3):<12.4f} {abs(t_z3_spin):<12.4f} {diff:<12.4e}")

    # The key question: are the singular values the same?
    print("\n--- Singular Value Comparison ---")
    S_plain_arr = S_plain.to_ndarray()
    S_z3_arr = S_z3.to_ndarray()

    print(f"Plain S: {S_plain_arr[:5]}")
    print(f"Z3 S:    {S_z3_arr[:5]}")

    diff_S = np.linalg.norm(S_plain_arr - S_z3_arr) / np.linalg.norm(S_plain_arr)
    print(f"||S_plain - S_z3|| / ||S_plain|| = {diff_S:.2e}")


def explore_fix():
    """
    Explore a potential fix: use a different criterion for CDL modes.

    Instead of trace, could use:
    1. The block-diagonal pattern of U
    2. The relation between q=1 and q=2 blocks
    3. Projection onto identity in each sector
    """
    print("\n" + "=" * 80)
    print("Exploring Potential Fix")
    print("=" * 80)

    pars_z3 = {
        'q': 3, 'relT': 1.0, 'gilt_eps': 1e-6,
        'cg_chis': [16], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': True,
    }

    A_z3 = get_initial_tensor_potts_relT(pars_z3)
    U_z3, S_z3 = get_envspec(A_z3, A_z3, pars_z3, where="S")

    arr = U_z3.to_ndarray()
    chi = arr.shape[0]
    n_modes = arr.shape[2]

    print("\nAlternative CDL mode criteria:")
    print(f"{'Mode':<6} {'|trace|':<10} {'|diag_sum|':<12} {'max_diag':<12} {'identity_proj':<15}")
    print("-" * 65)

    for i in range(min(10, n_modes)):
        U_i = arr[:, :, i]

        # Standard trace
        tr = np.trace(U_i)

        # Sum of diagonal magnitudes (different from trace if complex)
        diag_sum = np.sum(np.diag(U_i))

        # Maximum diagonal element
        max_diag = np.max(np.abs(np.diag(U_i)))

        # Projection onto identity matrix (normalized)
        I = np.eye(chi) / np.sqrt(chi)
        identity_proj = np.abs(np.sum(U_i.conj() * I))

        print(f"{i:<6} {abs(tr):<10.4f} {abs(diag_sum):<12.4f} {max_diag:<12.4f} {identity_proj:<15.4f}")

    print("""
The issue is that for Z3, the modes that should be CDL modes
(identity-like in spin basis) appear differently in charge basis
due to the block-diagonal structure.

Potential fix approaches:
1. Use |trace| / sqrt(chi) as the criterion (normalized)
2. Compute trace in spin basis (DFT transform)
3. Use identity projection instead of trace
4. Apply GILT in spin basis (as originally planned)
""")


if __name__ == "__main__":
    compare_mode_structure()
    explore_fix()
