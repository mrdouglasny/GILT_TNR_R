#!/usr/bin/env python3
"""
Debug the Z3 trace mismatch.

The single-site trace (partition function) differs by factor of 2.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D_Potts import get_initial_tensor_potts, get_initial_tensor_potts_z3
from tensors import Tensor, TensorZ3

def main():
    print("=" * 70)
    print("Debug Z3 Trace Mismatch")
    print("=" * 70)

    pars = {'q': 3, 'beta': np.log(1 + np.sqrt(3))}

    # Get tensors
    A_plain = get_initial_tensor_potts(pars)
    A_z3 = get_initial_tensor_potts_z3(pars)

    arr_plain = A_plain.to_ndarray()
    arr_z3 = A_z3.to_ndarray()

    # Plain tensor trace
    trace_plain = np.einsum('aaaa', arr_plain)
    print(f"\nPlain tensor trace: {trace_plain:.6f}")

    # Z3 tensor trace using numpy einsum
    trace_z3_numpy = np.einsum('aaaa', arr_z3)
    print(f"Z3 tensor trace (numpy): {trace_z3_numpy:.6f}")

    # Z3 tensor trace using TensorZ3 method (if available)
    # Let's check what the trace method does
    print(f"\n--- Detailed analysis ---")

    # Check the diagonal elements
    print(f"\nDiagonal elements T[a,a,a,a]:")
    print(f"  Plain: T[0,0,0,0]={arr_plain[0,0,0,0]:.4f}, T[1,1,1,1]={arr_plain[1,1,1,1]:.4f}, T[2,2,2,2]={arr_plain[2,2,2,2]:.4f}")
    print(f"  Z3:    T[0,0,0,0]={arr_z3[0,0,0,0]:.4f}, T[1,1,1,1]={arr_z3[1,1,1,1]:.4f}, T[2,2,2,2]={arr_z3[2,2,2,2]:.4f}")

    sum_plain = arr_plain[0,0,0,0] + arr_plain[1,1,1,1] + arr_plain[2,2,2,2]
    sum_z3 = arr_z3[0,0,0,0] + arr_z3[1,1,1,1] + arr_z3[2,2,2,2]
    print(f"  Sum of diagonals: plain={sum_plain:.4f}, z3={sum_z3:.4f}")

    # The DFT should preserve the trace!
    # Let's verify this manually
    print("\n--- DFT properties check ---")

    omega = np.exp(2j * np.pi / 3)
    F = np.array([
        [1, 1, 1],
        [1, omega, omega**2],
        [1, omega**2, omega]
    ]) / np.sqrt(3)

    print(f"F matrix:\n{F}")
    print(f"F @ F† = I check: {np.allclose(F @ F.T.conj(), np.eye(3))}")

    # For a single index, DFT should preserve the trace
    # Because Tr(F X F†) = Tr(X) for unitary F
    v = np.array([1, 2, 3])
    v_transformed = F @ v
    print(f"\nVector v = {v}")
    print(f"F @ v = {v_transformed}")
    print(f"Sum(v) = {sum(v)}, Sum(F@v) = {sum(v_transformed):.4f}")

    # The issue: For FOUR indices, we use F⊗F⊗F†⊗F†
    # This is NOT the same as F⊗4 or F†⊗4

    # Let's compute the expected trace after transform
    print("\n--- 4-index transform analysis ---")

    # T_charge[i,j,k,l] = sum_{a,b,c,d} F[i,a] F[j,b] F*[k,c] F*[l,d] T_spin[a,b,c,d]
    # Trace: sum_i T_charge[i,i,i,i] = sum_i sum_{a,b,c,d} F[i,a] F[i,b] F*[i,c] F*[i,d] T_spin[a,b,c,d]

    # Let's compute this explicitly
    trace_check = 0
    for i in range(3):
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    for d in range(3):
                        trace_check += F[i,a] * F[i,b] * np.conj(F[i,c]) * np.conj(F[i,d]) * arr_plain[a,b,c,d]

    print(f"Explicit trace computation: {trace_check:.6f}")

    # Compare with einsum
    print(f"numpy einsum trace: {trace_z3_numpy:.6f}")

    # What's sum_i F[i,a]*F[i,b]*F*[i,c]*F*[i,d]?
    print("\n--- Check sum_i F[i,a]*F[i,b]*F*[i,c]*F*[i,d] ---")

    # For diagonal a=b=c=d:
    for a in range(3):
        val = sum(F[i,a] * F[i,a] * np.conj(F[i,a]) * np.conj(F[i,a]) for i in range(3))
        print(f"  a=b=c=d={a}: {val:.6f}")

    # For general case:
    # sum_i F[i,a]F[i,b]F*[i,c]F*[i,d]
    # = sum_i |F[i,a]|^2 |F[i,b]|^2 if a=c and b=d
    # But F[i,a] is complex, so this is more complicated

    # Actually: F[i,a] = omega^{ia}/sqrt(3)
    # sum_i omega^{ia} omega^{ib} omega^{-ic} omega^{-id} / 9
    # = sum_i omega^{i(a+b-c-d)} / 9
    # = delta_{(a+b-c-d) mod 3, 0} * 3 / 9 = delta / 3

    print("\n--- Charge conservation check ---")
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    val = sum(F[i,a] * F[i,b] * np.conj(F[i,c]) * np.conj(F[i,d]) for i in range(3))
                    if abs(val) > 1e-10:
                        charge = (a + b - c - d) % 3
                        print(f"  ({a},{b},{c},{d}): charge={(a+b-c-d)}mod3={charge}, val={val:.4f}")

    # The issue: when we sum over diagonal (a=b=c=d), the charge is always 0
    # But the weight is 1/3 instead of 1!
    print("\n--- This explains the factor of 1/3! ---")
    print("When we trace T_charge, we get:")
    print("  Tr(T_charge) = sum_a (1/3) * T_spin[a,a,a,a] = (1/3) * Tr(T_spin)")
    print(f"  Expected: {trace_plain / 3:.6f}")
    print(f"  Got:      {trace_z3_numpy.real:.6f}")
    print(f"  Ratio:    {trace_plain / trace_z3_numpy.real:.6f}")

if __name__ == "__main__":
    main()
