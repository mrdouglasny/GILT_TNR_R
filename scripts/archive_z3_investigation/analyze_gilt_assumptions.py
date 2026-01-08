#!/usr/bin/env python3
"""
Analyze GILT assumptions: Why does Z2 work but Z3 fail?

Key question: What property of the environment tensor E differs?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D import get_envspec
from GiltTNR2D_Potts import get_initial_tensor_potts_relT
from tensors import Tensor, TensorZ2, TensorZ3


def get_ising_tensor(beta, use_z2=False):
    """Get 2D Ising tensor at inverse temperature beta."""
    # Standard construction
    W = np.array([[np.sqrt(np.cosh(beta)), np.sqrt(np.sinh(beta))],
                  [np.sqrt(np.cosh(beta)), -np.sqrt(np.sinh(beta))]])

    delta = np.zeros((2, 2, 2, 2))
    for i in range(2):
        delta[i, i, i, i] = 1

    T = np.einsum('ia,jb,kc,ld,abcd->ijkl', W, W, W, W, delta)

    if use_z2:
        # Use TensorZ2 with proper initialization
        try:
            return TensorZ2.from_ndarray(T, shape=T.shape)
        except:
            # If TensorZ2 doesn't work, just use plain
            print("  (TensorZ2 creation failed, using plain)")
            return Tensor.from_ndarray(T)
    else:
        return Tensor.from_ndarray(T)


def analyze_environment(name, A, pars):
    """Analyze the environment tensor structure."""
    print(f"\n{'='*70}")
    print(f"Environment Analysis: {name}")
    print(f"{'='*70}")

    U, S = get_envspec(A, A, pars, where="S")

    arr_U = U.to_ndarray()
    arr_S = S.to_ndarray()

    chi = arr_U.shape[0]

    print(f"\nTensor type: {type(A).__name__}")
    print(f"U shape: {arr_U.shape}")
    print(f"Bond dimension χ: {chi}")

    # Check if U matrices are block-diagonal
    print(f"\n--- U Matrix Structure ---")

    for i in range(min(3, arr_U.shape[2])):
        U_i = arr_U[:, :, i]
        # Check off-diagonal norm
        diag = np.diag(U_i)
        off_diag = U_i - np.diag(diag)
        off_diag_norm = np.linalg.norm(off_diag)
        total_norm = np.linalg.norm(U_i)

        print(f"\n  U[{i}] (S={arr_S[i]:.2e}):")
        print(f"    ||off-diagonal|| / ||total|| = {off_diag_norm/total_norm:.4f}")

        # For small matrices, show structure
        if chi <= 4:
            print(f"    Matrix:\n{np.abs(U_i)}")

    # Compute traces
    t = ncon(U, [1, 1, -1]).to_ndarray()

    print(f"\n--- Trace Structure ---")
    print(f"  t = ncon(U, [1,1,-1])")
    print(f"  Non-zero traces (|t| > 0.01):")
    for i, ti in enumerate(t):
        if abs(ti) > 0.01:
            print(f"    t[{i}] = {ti:.4f} (S={arr_S[i]:.2e})")

    # Count how many modes have significant trace
    n_nonzero = np.sum(np.abs(t) > 0.01)
    print(f"\n  Number of modes with |t| > 0.01: {n_nonzero}")

    # Key metric: What fraction of modes contribute to filtering?
    total_trace_contribution = np.sum(np.abs(t))
    print(f"  Total |t| sum: {total_trace_contribution:.4f}")
    print(f"  Max possible (√χ per mode × n_modes): {np.sqrt(chi) * len(t):.4f}")

    return n_nonzero, total_trace_contribution


def main():
    print("=" * 70)
    print("GILT Assumption Analysis: Z2 vs Z3")
    print("=" * 70)

    # Ising at criticality
    beta_c_ising = 0.5 * np.log(1 + np.sqrt(2))

    print(f"\n\n{'#'*70}")
    print("PART 1: ISING MODEL (Z2 symmetry)")
    print(f"{'#'*70}")

    pars_ising_plain = {
        'gilt_eps': 1e-6,
        'cg_chis': [16],
        'cg_eps': 1e-10,
        'verbosity': 0,
    }

    A_ising_plain = get_ising_tensor(beta_c_ising, use_z2=False)
    A_ising_z2 = get_ising_tensor(beta_c_ising, use_z2=True)

    n1, t1 = analyze_environment("Ising Plain", A_ising_plain, pars_ising_plain)
    n2, t2 = analyze_environment("Ising Z2", A_ising_z2, pars_ising_plain)

    print(f"\n--- Ising Comparison ---")
    print(f"  Plain: {n1} non-zero modes, total |t| = {t1:.4f}")
    print(f"  Z2:    {n2} non-zero modes, total |t| = {t2:.4f}")

    # Potts at criticality
    print(f"\n\n{'#'*70}")
    print("PART 2: 3-STATE POTTS MODEL (Z3 symmetry)")
    print(f"{'#'*70}")

    pars_potts_plain = {
        'q': 3,
        'relT': 1.0,
        'gilt_eps': 1e-6,
        'cg_chis': [16],
        'cg_eps': 1e-10,
        'verbosity': 0,
        'symmetry_tensors': False,
    }
    pars_potts_z3 = dict(pars_potts_plain)
    pars_potts_z3['symmetry_tensors'] = True

    A_potts_plain = get_initial_tensor_potts_relT(pars_potts_plain)
    A_potts_z3 = get_initial_tensor_potts_relT(pars_potts_z3)

    n3, t3 = analyze_environment("Potts Plain", A_potts_plain, pars_potts_plain)
    n4, t4 = analyze_environment("Potts Z3", A_potts_z3, pars_potts_z3)

    print(f"\n--- Potts Comparison ---")
    print(f"  Plain: {n3} non-zero modes, total |t| = {t3:.4f}")
    print(f"  Z3:    {n4} non-zero modes, total |t| = {t4:.4f}")

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    print("""
Key Question: Why does Z2 work but Z3 fail?

Hypothesis 1: Number of non-zero trace modes
  - If Z2 and plain have same number, but Z3 has different → Z3 problem

Hypothesis 2: Distribution of trace values
  - If Z3 distributes the trace across more modes → filtering is different

Hypothesis 3: Block structure of U matrices
  - Z2 has 2 blocks, Z3 has 3 blocks
  - More blocks → more ways for trace to be distributed incorrectly
""")

    print(f"\nResults:")
    print(f"  Ising Plain: {n1} modes")
    print(f"  Ising Z2:    {n2} modes")
    print(f"  Potts Plain: {n3} modes")
    print(f"  Potts Z3:    {n4} modes")

    if n1 == n2 and n3 != n4:
        print(f"\n✓ Hypothesis 1 SUPPORTED: Z2 matches plain, Z3 doesn't")
    elif n1 != n2:
        print(f"\n✗ Hypothesis 1 NOT SUPPORTED: Even Z2 differs from plain")


if __name__ == "__main__":
    main()
