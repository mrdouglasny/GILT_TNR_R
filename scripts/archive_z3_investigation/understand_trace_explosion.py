#!/usr/bin/env python3
"""
Understand why Z3 develops so many more "identity-like" modes.

Key observation from previous analysis:
- Step 0: Both have 2 non-zero trace modes
- Step 2: Plain has 24, Z3 has 67 non-zero trace modes!

Why does Z3 "explode" in the number of modes with non-zero trace?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D import gilttnr_step, get_envspec
from GiltTNR2D_Potts import get_initial_tensor_potts_relT


def analyze_U_block_structure(U, name):
    """Analyze the block structure of U matrices."""
    arr_U = U.to_ndarray()
    chi = arr_U.shape[0]
    n_modes = arr_U.shape[2]

    print(f"\n{name}:")
    print(f"  χ = {chi}, n_modes = {n_modes}")

    # For each mode, check if it's block-diagonal
    n_diagonal = 0
    n_off_diagonal = 0

    for i in range(n_modes):
        U_i = arr_U[:, :, i]
        diag = np.diag(np.diag(U_i))
        off_diag_ratio = np.linalg.norm(U_i - diag) / (np.linalg.norm(U_i) + 1e-10)

        if off_diag_ratio < 0.1:
            n_diagonal += 1
        else:
            n_off_diagonal += 1

    print(f"  Diagonal-like modes (off-diag < 10%): {n_diagonal}")
    print(f"  Off-diagonal modes: {n_off_diagonal}")

    # Key insight: Diagonal matrices always have non-zero trace!
    # For a diagonal matrix D, trace(D) = sum of diagonal elements
    # For block-diagonal Z3, each mode lives in one sector, so trace ≠ 0

    # Count how many modes have significant trace
    t = ncon(U, [1, 1, -1]).to_ndarray()
    n_nonzero_trace = np.sum(np.abs(t) > 0.1)

    print(f"  Modes with |trace| > 0.1: {n_nonzero_trace}")

    # Check correlation between diagonal structure and trace
    print(f"\n  Correlation analysis:")
    diag_with_trace = 0
    diag_without_trace = 0
    offdiag_with_trace = 0
    offdiag_without_trace = 0

    for i in range(n_modes):
        U_i = arr_U[:, :, i]
        diag = np.diag(np.diag(U_i))
        off_diag_ratio = np.linalg.norm(U_i - diag) / (np.linalg.norm(U_i) + 1e-10)
        is_diagonal = off_diag_ratio < 0.1
        has_trace = np.abs(t[i]) > 0.1

        if is_diagonal and has_trace:
            diag_with_trace += 1
        elif is_diagonal and not has_trace:
            diag_without_trace += 1
        elif not is_diagonal and has_trace:
            offdiag_with_trace += 1
        else:
            offdiag_without_trace += 1

    print(f"    Diagonal + trace:      {diag_with_trace}")
    print(f"    Diagonal + no trace:   {diag_without_trace}")
    print(f"    Off-diag + trace:      {offdiag_with_trace}")
    print(f"    Off-diag + no trace:   {offdiag_without_trace}")


def main():
    print("=" * 70)
    print("Understanding the Trace Explosion in Z3")
    print("=" * 70)

    chi = 16
    gilt_eps = 1e-6

    pars_plain = {
        'q': 3, 'relT': 1.0, 'gilt_eps': gilt_eps,
        'cg_chis': [chi], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': False,
    }
    pars_z3 = dict(pars_plain)
    pars_z3['symmetry_tensors'] = True

    # Get tensors at step 2
    A_plain = get_initial_tensor_potts_relT(pars_plain)
    A_z3 = get_initial_tensor_potts_relT(pars_z3)

    # Run to step 2
    for _ in range(2):
        A_plain, _ = gilttnr_step(A_plain, 0.0, pars_plain)
        A_z3, _ = gilttnr_step(A_z3, 0.0, pars_z3)

    # Analyze U structure
    U_plain, _ = get_envspec(A_plain, A_plain, pars_plain, where="S")
    U_z3, _ = get_envspec(A_z3, A_z3, pars_z3, where="S")

    analyze_U_block_structure(U_plain, "Plain at Step 2")
    analyze_U_block_structure(U_z3, "Z3 at Step 2")

    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
For block-diagonal tensors (Z3), the U matrices from environment SVD
tend to be more diagonal. And diagonal matrices ALWAYS have non-zero trace!

For a diagonal matrix D = diag(d1, d2, ..., dn):
  trace(D) = d1 + d2 + ... + dn ≠ 0 (in general)

For a matrix with off-diagonal elements, trace can be zero even if
the matrix is non-zero:
  trace([[0, 1], [1, 0]]) = 0

So Z3's block-diagonal structure naturally produces MORE modes with
non-zero trace, which GILT interprets as "identity-like" CDL modes
that should be filtered.

This is the FUNDAMENTAL ISSUE: GILT's trace criterion assumes
off-diagonal matrices, but Z3 produces diagonal matrices.
""")


if __name__ == "__main__":
    main()
