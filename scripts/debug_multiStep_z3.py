#!/usr/bin/env python3
"""
Track Z3 tensor structure through multiple Gilt-TNR steps.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from tensors import Tensor, TensorZ2, TensorZ3
from GiltTNR2D import gilttnr_step
from GiltTNR2D_Potts import get_initial_tensor_potts_relT, get_scaldims_potts
from GiltTNR2D_Ising_benchmarks import get_initial_tensor, get_scaldims

def analyze_step(A, step, name):
    """Analyze tensor after each step."""
    print(f"\n--- {name} Step {step} ---")
    print(f"  Type: {type(A).__name__}")

    if hasattr(A, 'qodulus'):
        print(f"  shape: {A.shape}")
        print(f"  qhape: {A.qhape}")
        print(f"  dirs: {A.dirs}")

        # Count charge-conserving sectors
        charge_ok = 0
        for k in A.sects.keys():
            dirs = A.dirs
            charge_sum = sum(d * q for d, q in zip(dirs, k))
            if A.qodulus:
                charge_sum %= A.qodulus
            if charge_sum == A.charge:
                charge_ok += 1
        print(f"  Charge-conserving sectors: {charge_ok}/{len(A.sects)}")
    else:
        arr = A.to_ndarray()
        print(f"  shape: {arr.shape}")

    return A

def main():
    print("=" * 70)
    print("Multi-Step Z3 vs Z2 Tracking")
    print("=" * 70)

    chi = 16
    n_steps = 5

    print(f"\nParameters: χ={chi}, n_steps={n_steps}")

    # ===== Ising Z2 =====
    print("\n" + "=" * 70)
    print("ISING Z2")
    print("=" * 70)

    pars_z2 = {
        'beta': np.log(1 + np.sqrt(2)) / 2,
        'gilt_eps': 1e-6,
        'cg_chis': [chi],
        'cg_eps': 1e-10,
        'verbosity': 0,
        'symmetry_tensors': True,
    }

    A_z2 = get_initial_tensor(pars_z2)
    log_z2 = 0.0
    analyze_step(A_z2, 0, "Z2")

    print(f"\n{'Step':>4} {'x_1':>10} {'x_2':>10} {'Type':>10}")
    print("-" * 40)

    sd = get_scaldims(A_z2, pars_z2)
    print(f"{0:>4} {sd[1]:>10.4f} {sd[2]:>10.4f} {type(A_z2).__name__:>10}")

    for step in range(1, n_steps + 1):
        A_z2, log_z2 = gilttnr_step(A_z2, log_z2, pars_z2)
        analyze_step(A_z2, step, "Z2")
        sd = get_scaldims(A_z2, pars_z2)
        print(f"{step:>4} {sd[1]:>10.4f} {sd[2]:>10.4f} {type(A_z2).__name__:>10}")

    # ===== Potts Z3 =====
    print("\n" + "=" * 70)
    print("POTTS Z3")
    print("=" * 70)

    pars_z3 = {
        'q': 3,
        'relT': 1.0,
        'gilt_eps': 1e-6,
        'cg_chis': [chi],
        'cg_eps': 1e-10,
        'verbosity': 0,
        'symmetry_tensors': True,
    }

    A_z3 = get_initial_tensor_potts_relT(pars_z3)
    log_z3 = 0.0
    analyze_step(A_z3, 0, "Z3")

    print(f"\n{'Step':>4} {'x_1':>10} {'x_3':>10} {'Type':>10}")
    print("-" * 40)

    sd = get_scaldims_potts(A_z3)
    print(f"{0:>4} {sd[1]:>10.4f} {sd[3]:>10.4f} {type(A_z3).__name__:>10}")

    for step in range(1, n_steps + 1):
        A_z3, log_z3 = gilttnr_step(A_z3, log_z3, pars_z3)
        analyze_step(A_z3, step, "Z3")
        sd = get_scaldims_potts(A_z3)
        print(f"{step:>4} {sd[1]:>10.4f} {sd[3]:>10.4f} {type(A_z3).__name__:>10}")

    # ===== Plain Potts for comparison =====
    print("\n" + "=" * 70)
    print("POTTS PLAIN (for comparison)")
    print("=" * 70)

    pars_plain = dict(pars_z3)
    pars_plain['symmetry_tensors'] = False

    A_plain = get_initial_tensor_potts_relT(pars_plain)
    log_plain = 0.0

    print(f"\n{'Step':>4} {'x_1':>10} {'x_3':>10}")
    print("-" * 30)

    sd = get_scaldims_potts(A_plain)
    print(f"{0:>4} {sd[1]:>10.4f} {sd[3]:>10.4f}")

    for step in range(1, n_steps + 1):
        A_plain, log_plain = gilttnr_step(A_plain, log_plain, pars_plain)
        sd = get_scaldims_potts(A_plain)
        print(f"{step:>4} {sd[1]:>10.4f} {sd[3]:>10.4f}")

    print("\n" + "=" * 70)
    print("CFT values: x_σ = 0.133, x_ε = 0.8")
    print("=" * 70)

if __name__ == "__main__":
    main()
