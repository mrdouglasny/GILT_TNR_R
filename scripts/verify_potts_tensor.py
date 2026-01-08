#!/usr/bin/env python3
"""
Verify Potts tensor construction - compare Z3 vs plain tensors.

Check:
1. Initial tensors match (up to basis transformation)
2. Z3 tensor has correct charge structure
3. Tensor elements are real (no spurious imaginary parts)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D_Potts import (
    get_initial_tensor_potts,
    get_initial_tensor_potts_z3,
    get_initial_tensor_potts_relT,
)
from tensors import Tensor, TensorZ3

def main():
    print("=" * 70)
    print("Potts Tensor Verification")
    print("=" * 70)

    pars = {'q': 3, 'beta': np.log(1 + np.sqrt(3))}  # Critical point

    # 1. Plain tensor
    print("\n1. Plain Tensor (spin basis)")
    print("-" * 70)
    A_plain = get_initial_tensor_potts(pars)
    arr_plain = A_plain.to_ndarray()
    print(f"Shape: {arr_plain.shape}")
    print(f"Type: {type(A_plain)}")
    print(f"Norm: {np.linalg.norm(arr_plain):.6f}")
    print(f"Max element: {np.max(np.abs(arr_plain)):.6f}")
    print(f"Is real: {np.allclose(arr_plain.imag, 0) if np.iscomplexobj(arr_plain) else True}")

    # 2. Z3 tensor
    print("\n2. TensorZ3 (charge basis)")
    print("-" * 70)
    A_z3 = get_initial_tensor_potts_z3(pars)
    arr_z3 = A_z3.to_ndarray()
    print(f"Shape: {arr_z3.shape}")
    print(f"Type: {type(A_z3)}")
    print(f"TensorZ3 shape: {A_z3.shape}")
    print(f"TensorZ3 qhape: {A_z3.qhape}")
    print(f"TensorZ3 dirs: {A_z3.dirs}")
    print(f"Norm: {np.linalg.norm(arr_z3):.6f}")
    print(f"Max element: {np.max(np.abs(arr_z3)):.6f}")
    print(f"Is real: {np.allclose(arr_z3.imag, 0) if np.iscomplexobj(arr_z3) else True}")
    if np.iscomplexobj(arr_z3):
        print(f"Max imag part: {np.max(np.abs(arr_z3.imag)):.2e}")

    # 3. Check that both give same partition function
    print("\n3. Partition Function Check")
    print("-" * 70)
    # Single site: trace over diagonal
    Z_single_plain = np.einsum('aaaa', arr_plain)
    Z_single_z3 = np.einsum('aaaa', arr_z3)
    print(f"Single-site trace (plain): {Z_single_plain:.6f}")
    print(f"Single-site trace (Z3):    {Z_single_z3:.6f}")
    print(f"Difference: {abs(Z_single_plain - Z_single_z3):.2e}")

    # 4. Check Z3 charge structure
    print("\n4. Z3 Charge Structure")
    print("-" * 70)
    print(f"Sectors in tensor: {list(A_z3.sects.keys())}")
    for k, v in A_z3.sects.items():
        charge = sum(A_z3.dirs[i] * A_z3.qhape[i][k[i]] for i in range(len(k)))
        charge_mod3 = charge % 3
        print(f"  Key {k}: shape={v.shape}, sum(dir*q)={charge} mod 3 = {charge_mod3}")
        if charge_mod3 != 0:
            print(f"    WARNING: Non-zero charge! This block should be zero.")

    # 5. Transform Z3 back to spin basis and compare
    print("\n5. Inverse Transform Check")
    print("-" * 70)
    omega = np.exp(2j * np.pi / 3)
    F = np.array([
        [1, 1, 1],
        [1, omega, omega**2],
        [1, omega**2, omega]
    ]) / np.sqrt(3)
    F_inv = F.T.conj()  # F is unitary, so F^{-1} = F†

    # Transform back: T_spin = F†⊗F†⊗F⊗F T_charge
    # (inverse of what we did to go from spin to charge)
    arr_back = ncon(
        (arr_z3, F_inv, F_inv, F, F),
        ([1, 2, 3, 4], [-1, 1], [-2, 2], [3, -3], [4, -4])
    )

    # Compare to original
    diff = np.linalg.norm(arr_plain - arr_back.real)
    print(f"||T_plain - F†⊗F†⊗F⊗F T_z3||: {diff:.2e}")
    if diff > 1e-10:
        print("  WARNING: Inverse transform doesn't match original!")
        print(f"  Plain: {arr_plain[0,0,0,:]}")
        print(f"  Back:  {arr_back.real[0,0,0,:]}")

    # 6. Check SVD spectrum comparison
    print("\n6. SVD Spectrum Comparison")
    print("-" * 70)
    # SVD of plain tensor (as matrix)
    mat_plain = arr_plain.reshape(9, 9)
    S_plain = np.linalg.svd(mat_plain, compute_uv=False)
    S_plain = S_plain / S_plain[0]
    print(f"Plain singular values (normalized): {S_plain[:6]}")

    # SVD of Z3 tensor
    mat_z3 = arr_z3.reshape(9, 9)
    S_z3 = np.linalg.svd(mat_z3, compute_uv=False)
    S_z3 = S_z3 / S_z3[0]
    print(f"Z3 singular values (normalized):    {S_z3[:6]}")

    # 7. Check element magnitudes
    print("\n7. Element Magnitudes")
    print("-" * 70)
    print("Plain tensor (spin basis):")
    for i in range(3):
        print(f"  T[{i},{i},{i},{i}] = {arr_plain[i,i,i,i]:.6f}")
    print(f"  T[0,0,1,1] = {arr_plain[0,0,1,1]:.6f}")
    print(f"  T[0,1,0,1] = {arr_plain[0,1,0,1]:.6f}")

    print("\nZ3 tensor (charge basis):")
    for i in range(3):
        print(f"  T[{i},{i},{i},{i}] = {arr_z3[i,i,i,i]:.6f}")
    print(f"  T[0,0,1,1] = {arr_z3[0,0,1,1]:.6f}")
    print(f"  T[0,1,0,1] = {arr_z3[0,1,0,1]:.6f}")


if __name__ == "__main__":
    main()
