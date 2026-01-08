#!/usr/bin/env python3
"""
GILT-TNR flow entirely in spin basis.

Strategy:
1. Start with Z3 tensor, transform to spin basis once at the beginning
2. Run entire GILT-TNR flow using plain tensors in spin basis
3. Compare results with plain tensor flow (should be identical)
4. Compare with standard Z3 flow (will differ due to GILT mode issue)

This avoids the complexity of transforming back and forth between bases
at each step, which requires handling unbalanced sector dimensions.
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


def z3_to_spin(T_z3):
    """
    Transform TensorZ3 to plain tensor in spin basis.

    For initial tensor (chi=3):
    T_spin[s0, s1, s2, s3] = sum_{q} F†[s0,q0] F†[s1,q1] F[s2,q2] F[s3,q3] T_z3[q0,q1,q2,q3]
    """
    arr = T_z3.to_ndarray()

    if arr.shape != (3, 3, 3, 3):
        raise ValueError(f"Expected (3,3,3,3) tensor, got {arr.shape}")

    F = get_dft_matrix(3)
    Fdag = F.conj().T

    # Transformation: charge → spin
    # In-legs (0,1): apply F† to go from charge to spin
    # Out-legs (2,3): apply F† then take conjugate = apply F
    # Convention: T_spin = F† ⊗ F† ⊗ T_charge ⊗ F ⊗ F
    result = np.einsum('ai,bj,ijkl,ck,dl->abcd', Fdag, Fdag, arr, F, F)

    return Tensor.from_ndarray(result)


def count_gilt_modes(A, pars, threshold=0.1):
    """Count modes with |trace| > threshold."""
    pars_plain = dict(pars)
    pars_plain['symmetry_tensors'] = False

    total_modes = 0
    high_trace_modes = 0

    for where in ["S", "E", "N", "W"]:
        U, S = get_envspec(A, A, pars_plain, where=where)
        t = ncon(U, [1, 1, -1]).to_ndarray()
        total_modes += len(t)
        high_trace_modes += np.sum(np.abs(t) > threshold)

    return high_trace_modes, total_modes


def compare_flows():
    """Compare three GILT-TNR flows:
    1. Plain tensor (standard)
    2. Z3 tensor in charge basis (has mode issue)
    3. Z3 initial → spin basis → plain flow (should match plain)
    """
    print("=" * 90)
    print("GILT-TNR Flow Comparison: Plain vs Z3-Charge vs Z3-to-Spin")
    print("=" * 90)

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

    # Initialize tensors
    A_plain = get_initial_tensor_potts_relT(pars_plain)
    A_z3 = get_initial_tensor_potts_relT(pars_z3)

    # Transform Z3 to spin basis (becomes plain tensor)
    A_spin = z3_to_spin(A_z3)

    # Verify initial equivalence
    arr_plain = A_plain.to_ndarray()
    arr_spin = A_spin.to_ndarray()
    initial_diff = np.linalg.norm(arr_plain - arr_spin) / np.linalg.norm(arr_plain)
    print(f"\nInitial ||plain - Z3→spin|| / ||plain|| = {initial_diff:.2e}")

    if initial_diff > 1e-10:
        print("WARNING: Initial tensors don't match!")
        return
    print("✓ Initial tensors match\n")

    log_p, log_z3, log_spin = 0.0, 0.0, 0.0

    print(f"{'Step':<6} {'||A_plain||':<14} {'||A_z3||':<14} {'||A_spin||':<14} "
          f"{'Z3/P':<10} {'Spin/P':<10} {'Z3 modes':<10} {'Spin modes':<10}")
    print("-" * 100)

    for step in range(1, n_steps + 1):
        # Plain tensor flow
        A_plain, log_p = gilttnr_step(A_plain, log_p, pars_plain)
        norm_p = np.linalg.norm(A_plain.to_ndarray())

        # Z3 charge basis flow
        try:
            A_z3, log_z3 = gilttnr_step(A_z3, log_z3, pars_z3)
            norm_z3 = np.linalg.norm(A_z3.to_ndarray())
        except Exception as e:
            print(f"  Z3 error at step {step}: {e}")
            norm_z3 = float('nan')

        # Spin basis flow (plain tensor, but started from Z3)
        A_spin, log_spin = gilttnr_step(A_spin, log_spin, pars_plain)
        norm_spin = np.linalg.norm(A_spin.to_ndarray())

        r_z3 = norm_z3 / norm_p if not np.isnan(norm_z3) else float('nan')
        r_spin = norm_spin / norm_p

        # Count high-trace modes
        try:
            modes_z3, _ = count_gilt_modes(A_z3, pars_z3)
        except:
            modes_z3 = float('nan')
        modes_spin, _ = count_gilt_modes(A_spin, pars_plain)

        print(f"{step:<6} {norm_p:<14.4e} {norm_z3:<14.4e} {norm_spin:<14.4e} "
              f"{r_z3:<10.4f} {r_spin:<10.4f} {modes_z3:<10} {modes_spin:<10}")

    print("\n" + "-" * 100)
    print("Key observations:")
    print("- 'Plain' and 'Spin' should track closely (same algorithm, same basis)")
    print("- 'Z3' diverges because GILT incorrectly identifies CDL modes in charge basis")
    print("- 'Z3 modes' >> 'Spin modes' confirms the mode count discrepancy")


def measure_scaling_dimensions():
    """Measure scaling dimensions using spin-basis flow."""
    print("\n" + "=" * 90)
    print("Scaling Dimension Measurement with Spin-Basis GILT-TNR")
    print("=" * 90)

    chi = 20  # Higher chi for better accuracy
    gilt_eps = 1e-6
    n_steps = 8

    pars = {
        'q': 3, 'relT': 1.0, 'gilt_eps': gilt_eps,
        'cg_chis': [chi], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': False,
    }
    pars_z3 = dict(pars)
    pars_z3['symmetry_tensors'] = True

    # Start from Z3, transform to spin basis
    A_z3 = get_initial_tensor_potts_relT(pars_z3)
    A = z3_to_spin(A_z3)
    log_fact = 0.0

    print(f"\n{'Step':<6} {'Norm':<12} {'log(Z)':<14} {'x_ε (from λ)':<14}")
    print("-" * 50)

    for step in range(1, n_steps + 1):
        A, log_fact = gilttnr_step(A, log_fact, pars)
        norm = np.linalg.norm(A.to_ndarray())

        # Extract transfer matrix eigenvalues for scaling dimensions
        # The tensor after RG has legs [W, S, E, N]
        # Transfer matrix: T[W,E; W',E'] = A[W,S,E,N] * A*[W',S,E',N] contracted on S,N
        arr = A.to_ndarray()
        chi_curr = arr.shape[0]

        # Reshape to matrix form and get eigenvalues
        # T = Tr_{S,N}(A ⊗ A*)
        T_mat = np.einsum('wsen,xsey->wxey', arr, arr.conj())
        T_mat = T_mat.reshape(chi_curr * chi_curr, chi_curr * chi_curr)

        # Get leading eigenvalues
        eigs = np.linalg.eigvals(T_mat)
        eigs_sorted = np.sort(np.abs(eigs))[::-1]

        if len(eigs_sorted) >= 2 and eigs_sorted[0] > 0:
            ratio = eigs_sorted[1] / eigs_sorted[0]
            if ratio > 0:
                x_eps = -np.log(ratio) / np.log(2)  # 2^n scaling
            else:
                x_eps = float('nan')
        else:
            x_eps = float('nan')

        print(f"{step:<6} {norm:<12.4e} {log_fact:<14.4f} {x_eps:<14.4f}")

    print(f"\nCFT target: x_ε = 0.8 (3-state Potts)")


if __name__ == "__main__":
    compare_flows()
    measure_scaling_dimensions()
