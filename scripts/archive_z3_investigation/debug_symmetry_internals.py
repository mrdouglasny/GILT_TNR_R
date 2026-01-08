#!/usr/bin/env python3
"""
Debug the internal symmetry structure of TensorZ2 vs TensorZ3 after Gilt-TNR.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from tensors import Tensor, TensorZ2, TensorZ3
from GiltTNR2D import gilttnr_step
from GiltTNR2D_Potts import get_initial_tensor_potts_relT
from GiltTNR2D_Ising_benchmarks import get_initial_tensor

def analyze_tensor_symmetry(A, name):
    """Analyze the internal symmetry structure of a tensor."""
    print(f"\n--- {name} ---")
    print(f"  Type: {type(A).__name__}")

    if hasattr(A, 'qodulus'):
        print(f"  qodulus: {A.qodulus}")
        print(f"  shape: {A.shape}")
        print(f"  qhape: {A.qhape}")
        print(f"  dirs: {A.dirs}")
        print(f"  charge: {A.charge}")
        print(f"  invar: {A.invar}")

        # Count sectors
        print(f"  Number of sectors: {len(A.sects)}")

        # Check if sectors are charge-conserving
        charge_ok = 0
        for k in A.sects.keys():
            # k is a tuple of charges
            dirs = A.dirs
            charge_sum = sum(d * q for d, q in zip(dirs, k))
            if A.qodulus:
                charge_sum %= A.qodulus
            if charge_sum == A.charge:
                charge_ok += 1
        print(f"  Charge-conserving sectors: {charge_ok}/{len(A.sects)}")

        # Show sector dimensions
        total_elements = 0
        for k, v in A.sects.items():
            total_elements += v.size
        print(f"  Total elements in sectors: {total_elements}")

        # Show first few sectors
        print(f"  First 5 sectors:")
        for i, (k, v) in enumerate(list(A.sects.items())[:5]):
            dirs = A.dirs
            charge_sum = sum(d * q for d, q in zip(dirs, k))
            if A.qodulus:
                charge_sum %= A.qodulus
            is_conserving = charge_sum == A.charge
            print(f"    {k}: shape={v.shape}, charge_sum={charge_sum} ({'✓' if is_conserving else '✗'})")

    else:
        print(f"  Plain tensor (no symmetry)")
        arr = A.to_ndarray()
        print(f"  shape: {arr.shape}")

def compare_z2_z3():
    """Compare Z2 and Z3 internal behavior after Gilt-TNR."""
    print("=" * 70)
    print("Comparing Z2 (Ising) vs Z3 (Potts) Internal Structure")
    print("=" * 70)

    chi = 12

    # Test Ising Z2
    pars_z2 = {
        'beta': np.log(1 + np.sqrt(2)) / 2,
        'gilt_eps': 1e-6,
        'cg_chis': [chi],
        'cg_eps': 1e-10,
        'verbosity': 0,
        'symmetry_tensors': True,
    }
    A_z2 = get_initial_tensor(pars_z2)
    analyze_tensor_symmetry(A_z2, "Ising Z2 Initial")

    A_z2_new, _ = gilttnr_step(A_z2, 0.0, pars_z2)
    analyze_tensor_symmetry(A_z2_new, "Ising Z2 After Step 1")

    # Test Potts Z3
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
    analyze_tensor_symmetry(A_z3, "Potts Z3 Initial")

    A_z3_new, _ = gilttnr_step(A_z3, 0.0, pars_z3)
    analyze_tensor_symmetry(A_z3_new, "Potts Z3 After Step 1")

    # Also test plain tensors for comparison
    print("\n" + "=" * 70)
    print("Plain Tensor Comparison")
    print("=" * 70)

    pars_plain_z2 = dict(pars_z2)
    pars_plain_z2['symmetry_tensors'] = False
    A_plain_z2 = get_initial_tensor(pars_plain_z2)
    A_plain_z2_new, _ = gilttnr_step(A_plain_z2, 0.0, pars_plain_z2)
    analyze_tensor_symmetry(A_plain_z2_new, "Ising Plain After Step 1")

    pars_plain_z3 = dict(pars_z3)
    pars_plain_z3['symmetry_tensors'] = False
    A_plain_z3 = get_initial_tensor_potts_relT(pars_plain_z3)
    A_plain_z3_new, _ = gilttnr_step(A_plain_z3, 0.0, pars_plain_z3)
    analyze_tensor_symmetry(A_plain_z3_new, "Potts Plain After Step 1")

    # Compare shapes
    print("\n" + "=" * 70)
    print("Shape Comparison")
    print("=" * 70)
    print(f"  Z2 tensor shape: {A_z2_new.shape}")
    print(f"  Z3 tensor shape: {A_z3_new.shape}")
    print(f"  Plain Ising shape: {A_plain_z2_new.to_ndarray().shape}")
    print(f"  Plain Potts shape: {A_plain_z3_new.to_ndarray().shape}")

    # Compare physical results
    print("\n" + "=" * 70)
    print("Scaling Dimensions Comparison")
    print("=" * 70)

    from GiltTNR2D_Ising_benchmarks import get_scaldims
    from GiltTNR2D_Potts import get_scaldims_potts

    sd_z2 = get_scaldims(A_z2_new, pars_z2)
    sd_plain_z2 = get_scaldims(A_plain_z2_new, pars_plain_z2)
    print(f"  Ising Z2:     x_1={sd_z2[1]:.4f}, x_2={sd_z2[2]:.4f}")
    print(f"  Ising Plain:  x_1={sd_plain_z2[1]:.4f}, x_2={sd_plain_z2[2]:.4f}")
    print(f"  Ising CFT:    x_σ=0.125, x_ε=1.0")

    sd_z3 = get_scaldims_potts(A_z3_new)
    sd_plain_z3 = get_scaldims_potts(A_plain_z3_new)
    print(f"  Potts Z3:     x_1={sd_z3[1]:.4f}, x_3={sd_z3[3]:.4f}")
    print(f"  Potts Plain:  x_1={sd_plain_z3[1]:.4f}, x_3={sd_plain_z3[3]:.4f}")
    print(f"  Potts CFT:    x_σ=0.133, x_ε=0.8")

    # Check Z2 vs Plain match
    z2_match = np.allclose(sd_z2[:5], sd_plain_z2[:5], rtol=0.01)
    z3_match = np.allclose(sd_z3[:5], sd_plain_z3[:5], rtol=0.01)
    print(f"\n  Z2 matches Plain (1% tol): {'✓' if z2_match else '✗'}")
    print(f"  Z3 matches Plain (1% tol): {'✓' if z3_match else '✗'}")

if __name__ == "__main__":
    compare_z2_z3()
