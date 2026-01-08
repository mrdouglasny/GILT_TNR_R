#!/usr/bin/env python3
"""
Debug the Z3 symmetric tensor construction issue.

The previous test showed:
- Trace(A @ A†) in spin basis: 126
- Trace(A @ A†) via TensorZ3: 42

That's a factor of 3 difference! Let's trace through the construction.
"""

import sys
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr/GiltTNR')
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr')

import numpy as np
from tensors.symmetrytensors import TensorZ3
from tensors.tensor import Tensor

def build_dft_matrix(N):
    """Build N×N DFT matrix."""
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega**(s*q) for q in range(N)] for s in range(N)]) / np.sqrt(N)
    return F

def transform_spin_to_charge(arr_spin, N=3):
    """Transform tensor from spin basis to charge basis."""
    F = build_dft_matrix(N)
    Fdag = F.conj().T
    return np.einsum('ai,bj,ijkl,kc,ld->abcd', F, F, arr_spin, Fdag, Fdag)

def create_z3_symmetric_tensor(N=3, values=None):
    """Create a Z3 symmetric tensor in spin basis."""
    if values is None:
        values = {0: 3.0, 1: 2.0, 2: 1.0}

    arr = np.zeros((N, N, N, N), dtype=complex)
    for s1 in range(N):
        for s2 in range(N):
            for s3 in range(N):
                for s4 in range(N):
                    if (s1 + s2) % N == (s3 + s4) % N:
                        total_charge = (s1 + s2) % N
                        arr[s1, s2, s3, s4] = values[total_charge]
    return arr

def debug_trace_issue():
    """Debug the trace discrepancy."""
    print("="*70)
    print("Debugging trace issue")
    print("="*70)

    N = 3
    values = {0: 3.0, 1: 2.0, 2: 1.0}

    # Create in spin basis
    arr_spin = create_z3_symmetric_tensor(N, values)

    print("Spin basis tensor:")
    print(f"  Shape: {arr_spin.shape}")
    print(f"  Non-zero elements: {np.count_nonzero(arr_spin)}")
    trace_spin = np.einsum('ijkl,ijkl->', arr_spin, arr_spin.conj())
    print(f"  Trace(A @ A†) = {trace_spin}")

    # List non-zero elements
    print("\n  Non-zero elements by charge sector:")
    for charge in range(3):
        count = 0
        val_sum = 0
        for s1 in range(N):
            for s2 in range(N):
                for s3 in range(N):
                    for s4 in range(N):
                        if (s1 + s2) % N == charge and (s3 + s4) % N == charge:
                            if abs(arr_spin[s1, s2, s3, s4]) > 1e-10:
                                count += 1
                                val_sum += abs(arr_spin[s1, s2, s3, s4])**2
        print(f"    Charge {charge}: {count} elements, sum of |val|² = {val_sum}")

    # Transform to charge basis
    arr_charge = transform_spin_to_charge(arr_spin, N)

    print("\nCharge basis tensor:")
    print(f"  Shape: {arr_charge.shape}")
    print(f"  Non-zero elements: {np.count_nonzero(np.abs(arr_charge) > 1e-10)}")
    trace_charge = np.einsum('ijkl,ijkl->', arr_charge, arr_charge.conj())
    print(f"  Trace(A @ A†) = {trace_charge}")

    # List non-zero elements
    print("\n  All non-zero elements (|val| > 0.1):")
    for q0 in range(N):
        for q1 in range(N):
            for q2 in range(N):
                for q3 in range(N):
                    val = arr_charge[q0, q1, q2, q3]
                    if abs(val) > 0.1:
                        conserved = "✓" if (q0 + q1) % N == (q2 + q3) % N else "✗"
                        print(f"    [{q0},{q1},{q2},{q3}] = {val:.4f} (charge conserved: {conserved})")

    # Now create TensorZ3
    print("\n" + "-"*50)
    print("Creating TensorZ3 from charge basis array:")
    print("-"*50)

    A_z3 = TensorZ3.from_ndarray(arr_charge, shape=[[1,1,1]]*4, dirs=[1,1,-1,-1])

    print(f"  TensorZ3 shape: {A_z3.shape}")
    print(f"  Number of sectors: {len(A_z3.sects)}")

    # Get back as array
    arr_back = A_z3.to_ndarray()
    trace_z3 = np.einsum('ijkl,ijkl->', arr_back, arr_back.conj())
    print(f"  Trace(A @ A†) from TensorZ3: {trace_z3}")

    # Compare
    diff = np.linalg.norm(arr_back - arr_charge) / np.linalg.norm(arr_charge)
    print(f"  Relative difference arr_back vs arr_charge: {diff:.2e}")

    print("\n" + "-"*50)
    print("Comparing specific elements:")
    print("-"*50)

    for q0 in range(N):
        for q1 in range(N):
            for q2 in range(N):
                for q3 in range(N):
                    orig = arr_charge[q0, q1, q2, q3]
                    back = arr_back[q0, q1, q2, q3]
                    if abs(orig) > 0.1 or abs(back) > 0.1:
                        if abs(orig - back) > 1e-10:
                            print(f"  [{q0},{q1},{q2},{q3}]: original={orig:.4f}, back={back:.4f}, DIFF!")
                        # else:
                        #     print(f"  [{q0},{q1},{q2},{q3}]: original={orig:.4f}, back={back:.4f}, OK")

def debug_my_test_tensor():
    """Debug the specific tensor used in minimal_truncation_test.py"""
    print("\n" + "="*70)
    print("Debugging the test tensor from minimal_truncation_test.py")
    print("="*70)

    N = 3

    # This is exactly what was done in minimal_truncation_test.py
    values = {0: 3.0, 1: 2.0, 2: 1.0}
    arr_spin = create_z3_symmetric_tensor(N, values)

    print("Original in spin basis:")
    trace_spin = np.einsum('ijkl,ijkl->', arr_spin, arr_spin.conj())
    print(f"  Trace(A @ A†) = {trace_spin}")

    # Transform to charge
    arr_charge = transform_spin_to_charge(arr_spin, N)
    trace_charge = np.einsum('ijkl,ijkl->', arr_charge, arr_charge.conj())
    print(f"After spin->charge: Trace = {trace_charge}")

    # Create TensorZ3 with the WRONG dirs (no dirs specified uses default)
    print("\n--- Testing different dirs ---")

    for dirs in [None, [1,1,1,1], [1,1,-1,-1], [-1,-1,1,1]]:
        if dirs is None:
            A_z3 = TensorZ3.from_ndarray(arr_charge, shape=[[1,1,1]]*4)
        else:
            A_z3 = TensorZ3.from_ndarray(arr_charge, shape=[[1,1,1]]*4, dirs=dirs)

        arr_back = A_z3.to_ndarray()
        trace = np.einsum('ijkl,ijkl->', arr_back, arr_back.conj())
        diff = np.linalg.norm(arr_back - arr_charge)
        print(f"  dirs={dirs}: trace={trace.real:.1f}, diff from arr_charge={diff:.2e}")

def check_the_actual_potts_tensor():
    """Check what the actual Potts tensor construction does."""
    print("\n" + "="*70)
    print("Checking actual Potts tensor")
    print("="*70)

    from GiltTNR2D_Potts import get_initial_tensor_potts, get_initial_tensor_potts_z3

    beta_c = np.log(1 + np.sqrt(3))

    A_plain = get_initial_tensor_potts({"beta": beta_c})
    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})

    arr_plain = A_plain.to_ndarray()
    arr_z3 = A_z3.to_ndarray()

    trace_plain = np.einsum('ijkl,ijkl->', arr_plain, arr_plain.conj())
    trace_z3 = np.einsum('ijkl,ijkl->', arr_z3, arr_z3.conj())

    print(f"Plain tensor trace: {trace_plain}")
    print(f"TensorZ3 tensor trace: {trace_z3}")
    print(f"Ratio: {trace_plain / trace_z3}")

    # Transform plain to charge basis
    F = build_dft_matrix(3)
    Fdag = F.conj().T
    arr_plain_charge = np.einsum('ai,bj,ijkl,kc,ld->abcd', F, F, arr_plain, Fdag, Fdag)

    trace_plain_charge = np.einsum('ijkl,ijkl->', arr_plain_charge, arr_plain_charge.conj())
    print(f"Plain in charge basis trace: {trace_plain_charge}")

    # Compare with Z3
    diff = np.linalg.norm(arr_plain_charge - arr_z3) / np.linalg.norm(arr_plain_charge)
    print(f"Difference between plain->charge and Z3: {diff:.2e}")

if __name__ == "__main__":
    debug_trace_issue()
    debug_my_test_tensor()
    check_the_actual_potts_tensor()
