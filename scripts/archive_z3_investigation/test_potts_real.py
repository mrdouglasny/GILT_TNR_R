#!/usr/bin/env python3
"""
Test if the issue is complex vs real tensors.

For Z₃, the charge basis has complex DFT. But after the transformation,
the tensor SHOULD be real (due to Z₃ invariance of the original tensor).

Let's verify this and try forcing it to be real.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from tensors import Tensor, TensorZ3

def main():
    beta_c = np.log(1 + np.sqrt(3))
    q = 3

    # Build vertex tensor
    W = np.ones((q, q))
    np.fill_diagonal(W, np.exp(beta_c))
    T_spin = np.einsum('ab,bc,cd,da->abcd', W, W, W, W)

    # Z₃ DFT
    omega = np.exp(2j * np.pi / 3)
    F = np.array([
        [1, 1, 1],
        [1, omega, omega**2],
        [1, omega**2, omega]
    ]) / np.sqrt(3)
    F_dg = F.T.conj()

    # Transform to charge basis
    T_charge = ncon(
        (T_spin, F, F, F_dg, F_dg),
        ([1, 2, 3, 4], [-1, 1], [-2, 2], [3, -3], [4, -4])
    )

    print("Charge basis tensor analysis:")
    print(f"  Max |real|: {np.max(np.abs(T_charge.real)):.6f}")
    print(f"  Max |imag|: {np.max(np.abs(T_charge.imag)):.6f}")
    print(f"  Ratio imag/real: {np.max(np.abs(T_charge.imag)) / np.max(np.abs(T_charge.real)):.2e}")

    # Check specific charge-conserving elements
    print("\nCharge-conserving elements (should be real or have small imag):")
    for key in [(0,0,0,0), (0,0,1,2), (0,0,2,1), (1,1,1,1), (2,2,2,2)]:
        val = T_charge[key]
        print(f"  T{key} = {val.real:+.4f} + {val.imag:+.4f}i")

    # The imaginary parts should be zero for a Z₃-symmetric tensor
    # Let's verify by checking if T_charge is Hermitian under some transformation

    # Check if taking real part changes the TM spectrum
    T_real = T_charge.real.copy()
    T_real[np.abs(T_real) < 1e-10] = 0

    # TM from complex
    TM_complex = np.einsum('abcd,cdeg->abeg', T_charge, T_charge).reshape(9, 9)
    eigs_complex = np.abs(np.linalg.eigvals(TM_complex))
    eigs_complex = -np.sort(-eigs_complex)

    # TM from real part only
    TM_real = np.einsum('abcd,cdeg->abeg', T_real, T_real).reshape(9, 9)
    eigs_real = np.abs(np.linalg.eigvals(TM_real))
    eigs_real = -np.sort(-eigs_real)

    print("\nTransfer matrix eigenvalues:")
    print(f"  Complex: {eigs_complex[:4]}")
    print(f"  Real:    {eigs_real[:4]}")

    # Test TensorZ3 with real array
    dim = [1, 1, 1]
    qim = [0, 1, 2]
    dirs = [1, 1, -1, -1]

    # Force to real
    T_charge_real_forced = T_charge.real + 0j  # Still complex dtype but zero imag
    T_charge_real_forced[np.abs(T_charge_real_forced) < 1e-10] = 0

    print("\nCreating TensorZ3 from real-forced array:")
    T_z3_real = TensorZ3.from_ndarray(
        T_charge_real_forced,
        shape=[dim]*4,
        qhape=[qim]*4,
        dirs=dirs,
        charge=0,
        invar=True
    )
    print(f"  Success! Type: {type(T_z3_real)}")
    print(f"  dtype: {T_z3_real.dtype}")

    # Now test with pure complex
    print("\nCreating TensorZ3 from full complex array:")
    T_z3_complex = TensorZ3.from_ndarray(
        T_charge,
        shape=[dim]*4,
        qhape=[qim]*4,
        dirs=dirs,
        charge=0,
        invar=True
    )
    print(f"  Success! Type: {type(T_z3_complex)}")
    print(f"  dtype: {T_z3_complex.dtype}")

    # Compare scaldims
    def compute_scaldims(A):
        transmat = ncon((A, A), [[3,-101,4,-1], [4,-102,3,-2]])
        es = transmat.eig([0,1], [2,3], hermitian=False)[0]
        es = es.to_ndarray()
        es = np.abs(es)
        es = -np.sort(-es)
        es[es == 0] += 1e-16
        log_es = np.log(es)
        log_es -= np.max(log_es)
        log_es /= -np.pi
        return log_es

    sd_real = compute_scaldims(T_z3_real)
    sd_complex = compute_scaldims(T_z3_complex)

    print("\nScaling dimensions comparison:")
    print(f"  Real-forced:    x_1={sd_real[1]:.4f}, x_2={sd_real[2]:.4f}, x_3={sd_real[3]:.4f}")
    print(f"  Full complex:   x_1={sd_complex[1]:.4f}, x_2={sd_complex[2]:.4f}, x_3={sd_complex[3]:.4f}")

if __name__ == "__main__":
    main()
