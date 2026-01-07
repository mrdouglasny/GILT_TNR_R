#!/usr/bin/env python3
"""
Trace through a single Gilt-TNR step to see where TensorZ3 diverges from plain Tensor.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from tensors import Tensor, TensorZ3
from GiltTNR2D import gilttnr_step
from GiltTNR2D_Potts import get_initial_tensor_potts_relT, get_scaldims_potts

def tensor_summary(A, name):
    """Print summary of tensor state."""
    arr = A.to_ndarray()
    print(f"  {name}: type={type(A).__name__}, shape={arr.shape}, "
          f"max={np.max(np.abs(arr)):.4f}, dtype={arr.dtype}")

def main():
    print("=" * 70)
    print("Tracing single Gilt-TNR step")
    print("=" * 70)

    chi = 12
    pars = {
        'q': 3,
        'relT': 1.0,
        'gilt_eps': 1e-6,
        'cg_chis': [chi],
        'cg_eps': 1e-10,
        'verbosity': 2,  # More verbose
    }

    # Build initial tensors
    pars_plain = dict(pars)
    pars_plain['symmetry_tensors'] = False
    A_plain = get_initial_tensor_potts_relT(pars_plain)

    pars_z3 = dict(pars)
    pars_z3['symmetry_tensors'] = True
    A_z3 = get_initial_tensor_potts_relT(pars_z3)

    print("\nInitial state:")
    tensor_summary(A_plain, "Plain")
    tensor_summary(A_z3, "Z3")

    sd_plain = get_scaldims_potts(A_plain)
    sd_z3 = get_scaldims_potts(A_z3)
    print(f"  Plain scaldims: x_1={sd_plain[1]:.4f}, x_3={sd_plain[3]:.4f}")
    print(f"  Z3 scaldims:    x_1={sd_z3[1]:.4f}, x_3={sd_z3[3]:.4f}")

    # Run one Gilt step with verbose output
    print("\n" + "=" * 70)
    print("Running Gilt-TNR step on Plain Tensor...")
    print("=" * 70)
    A_plain_new, log_plain = gilttnr_step(A_plain, 0.0, pars_plain)

    print("\n" + "=" * 70)
    print("Running Gilt-TNR step on TensorZ3...")
    print("=" * 70)
    A_z3_new, log_z3 = gilttnr_step(A_z3, 0.0, pars_z3)

    print("\n" + "=" * 70)
    print("After one step:")
    print("=" * 70)
    tensor_summary(A_plain_new, "Plain")
    tensor_summary(A_z3_new, "Z3")

    sd_plain_new = get_scaldims_potts(A_plain_new)
    sd_z3_new = get_scaldims_potts(A_z3_new)
    print(f"  Plain scaldims: x_1={sd_plain_new[1]:.4f}, x_3={sd_plain_new[3]:.4f}")
    print(f"  Z3 scaldims:    x_1={sd_z3_new[1]:.4f}, x_3={sd_z3_new[3]:.4f}")

    # Compare arrays
    arr_plain = A_plain_new.to_ndarray()
    arr_z3 = A_z3_new.to_ndarray()

    print(f"\n  Array comparison:")
    print(f"    Plain shape: {arr_plain.shape}")
    print(f"    Z3 shape:    {arr_z3.shape}")

    if arr_plain.shape == arr_z3.shape:
        diff = np.max(np.abs(arr_plain - arr_z3))
        print(f"    Max diff:    {diff:.4e}")
    else:
        print(f"    Shapes differ!")

    # Check Z3 charge conservation after Gilt
    print("\n  Z3 charge conservation check after Gilt:")
    non_zero = 0
    charge_conserved = 0
    for idx in np.ndindex(*arr_z3.shape):
        if np.abs(arr_z3[idx]) > 1e-10:
            non_zero += 1
            # For dirs [1,1,-1,-1], conservation: sum(d*q) mod 3 = 0
            dirs = [1, 1, -1, -1]
            charge_sum = sum(d * q for d, q in zip(dirs, idx))
            if charge_sum % 3 == 0:
                charge_conserved += 1

    print(f"    Non-zero elements: {non_zero}")
    print(f"    Charge-conserved:  {charge_conserved}")
    if non_zero > 0:
        print(f"    Fraction:          {charge_conserved/non_zero*100:.1f}%")

    # Run second step
    print("\n" + "=" * 70)
    print("Running second Gilt-TNR step on TensorZ3...")
    print("=" * 70)
    A_z3_2, log_z3_2 = gilttnr_step(A_z3_new, log_z3, pars_z3)

    print("\nAfter two steps:")
    tensor_summary(A_z3_2, "Z3")
    sd_z3_2 = get_scaldims_potts(A_z3_2)
    print(f"  Z3 scaldims: x_1={sd_z3_2[1]:.4f}, x_3={sd_z3_2[3]:.4f}")

    arr_z3_2 = A_z3_2.to_ndarray()
    print("\n  Z3 charge conservation after step 2:")
    non_zero = 0
    charge_conserved = 0
    for idx in np.ndindex(*arr_z3_2.shape):
        if np.abs(arr_z3_2[idx]) > 1e-10:
            non_zero += 1
            dirs = [1, 1, -1, -1]
            charge_sum = sum(d * q for d, q in zip(dirs, idx))
            if charge_sum % 3 == 0:
                charge_conserved += 1

    print(f"    Non-zero elements: {non_zero}")
    print(f"    Charge-conserved:  {charge_conserved}")
    if non_zero > 0:
        print(f"    Fraction:          {charge_conserved/non_zero*100:.1f}%")

if __name__ == "__main__":
    main()
