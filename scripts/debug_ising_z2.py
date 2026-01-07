#!/usr/bin/env python3
"""Check how Ising TensorZ2 works - as a reference for Potts TensorZ3."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from tensors import Tensor, TensorZ2

def main():
    # Ising critical point
    beta = np.log(1 + np.sqrt(2)) / 2
    print(f"Ising at β_c = {beta:.6f}")

    # Build Ising tensor (spin basis)
    hamiltonian = np.array([[-1, 1], [1, -1]])
    boltz = np.exp(-beta * hamiltonian)
    T_spin = np.einsum('ab,bc,cd,da->abcd', boltz, boltz, boltz, boltz)
    print(f"\nSpin basis tensor:")
    print(f"  T[0,0,0,0] = {T_spin[0,0,0,0]:.6f}")
    print(f"  T[0,0,1,1] = {T_spin[0,0,1,1]:.6f}")
    print(f"  T[0,1,0,1] = {T_spin[0,1,0,1]:.6f}")

    # Hadamard transform (Z₂ DFT)
    u = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    u_dg = u.T.conj()  # For Z₂, u = u†

    # Transform to charge basis
    T_charge = ncon((T_spin, u, u, u_dg, u_dg),
                    ([1, 2, 3, 4], [-1, 1], [-2, 2], [3, -3], [4, -4]))

    print(f"\nCharge basis tensor:")
    print(f"  T[0,0,0,0] = {T_charge[0,0,0,0]:.6f}")
    print(f"  T[0,0,1,1] = {T_charge[0,0,1,1]:.6f}")
    print(f"  T[0,1,0,1] = {T_charge[0,1,0,1]:.6f}")
    print(f"  T[1,1,0,0] = {T_charge[1,1,0,0]:.6f}")  # Should be 0 (charge not conserved)

    # Check charge conservation
    print("\nCharge conservation check:")
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    val = T_charge[i, j, k, l]
                    conserved = (i + j) % 2 == (k + l) % 2
                    if np.abs(val) > 1e-10:
                        status = "✓" if conserved else "✗"
                        print(f"  [{i},{j},{k},{l}]: {val:.4f} {status}")

    # Create TensorZ2
    dim, qim = [1, 1], [0, 1]
    T_z2 = TensorZ2.from_ndarray(T_charge, shape=[dim]*4, qhape=[qim]*4, dirs=[1, 1, -1, -1])

    # Extract and compare
    T_z2_arr = T_z2.to_ndarray()
    print(f"\nRoundtrip check:")
    print(f"  Input T_charge[0,0,0,0] = {T_charge[0,0,0,0]:.6f}")
    print(f"  TensorZ2 arr[0,0,0,0]   = {T_z2_arr[0,0,0,0]:.6f}")
    print(f"  Max diff: {np.max(np.abs(T_charge - T_z2_arr)):.2e}")

    # Compare scaling dimensions
    print("\nScaling dimensions:")

    # Plain Tensor
    T_plain = Tensor.from_ndarray(T_charge)
    transmat_plain = ncon((T_plain, T_plain), [[3,-101,4,-1], [4,-102,3,-2]])
    arr_tm_plain = transmat_plain.to_ndarray().reshape(4, 4)
    eigs_plain = np.abs(np.linalg.eigvals(arr_tm_plain))
    eigs_plain = -np.sort(-eigs_plain)
    scaldims_plain = np.log(eigs_plain) / (-np.pi)
    scaldims_plain -= scaldims_plain[0]
    print(f"  Plain:    x_1={scaldims_plain[1]:.4f}, x_2={scaldims_plain[2]:.4f}")

    # TensorZ2
    transmat_z2 = ncon((T_z2, T_z2), [[3,-101,4,-1], [4,-102,3,-2]])
    arr_tm_z2 = transmat_z2.to_ndarray().reshape(4, 4)
    eigs_z2 = np.abs(np.linalg.eigvals(arr_tm_z2))
    eigs_z2 = -np.sort(-eigs_z2)
    scaldims_z2 = np.log(eigs_z2) / (-np.pi)
    scaldims_z2 -= scaldims_z2[0]
    print(f"  Z2:       x_1={scaldims_z2[1]:.4f}, x_2={scaldims_z2[2]:.4f}")
    print(f"\n  Expected: x_σ = 1/8 = 0.125")

if __name__ == "__main__":
    main()
