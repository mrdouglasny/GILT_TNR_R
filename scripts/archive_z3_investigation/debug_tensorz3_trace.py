#!/usr/bin/env python3
"""
Debug TensorZ3 trace and normalization issues.

Key finding: Trace differs by factor of 3 between TensorZ3 and plain!
This suggests a normalization issue in how TensorZ3 stores/returns data.
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

def test_simple_tensor():
    """Test with a very simple tensor."""
    print("="*70)
    print("Test: Simple tensor")
    print("="*70)

    N = 3

    # Create the simplest possible Z3 tensor: identity-like
    # A[i,j,k,l] = δ_{i,k} δ_{j,l} in spin basis
    arr_spin = np.zeros((N, N, N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            arr_spin[i, j, i, j] = 1.0

    print(f"Created simple tensor in spin basis")
    print(f"Non-zero elements: {np.count_nonzero(arr_spin)}")
    print(f"Trace(A @ A†) = {np.einsum('ijkl,ijkl->', arr_spin, arr_spin.conj())}")

    # Transform to charge basis
    F = build_dft_matrix(N)
    Fdag = F.conj().T
    arr_charge = np.einsum('ai,bj,ijkl,kc,ld->abcd', F, F, arr_spin, Fdag, Fdag)

    print(f"\nIn charge basis:")
    print(f"Non-zero elements: {np.count_nonzero(np.abs(arr_charge) > 1e-10)}")
    print(f"Trace(A @ A†) = {np.einsum('ijkl,ijkl->', arr_charge, arr_charge.conj())}")

    # Now check what TensorZ3.from_ndarray does
    print("\n--- TensorZ3 analysis ---")

    A_z3 = TensorZ3.from_ndarray(arr_charge, shape=[[1,1,1]]*4, dirs=[1,1,-1,-1])

    print(f"TensorZ3 shape: {A_z3.shape}")
    print(f"TensorZ3 sects keys: {list(A_z3.sects.keys())[:10]}...")

    # Get array back
    arr_back = A_z3.to_ndarray()
    print(f"\nto_ndarray shape: {arr_back.shape}")
    print(f"Trace(back @ back†) = {np.einsum('ijkl,ijkl->', arr_back, arr_back.conj())}")

    # Compare element by element
    diff = np.linalg.norm(arr_back - arr_charge) / np.linalg.norm(arr_charge)
    print(f"Relative difference arr_back vs arr_charge: {diff:.2e}")

def test_diagonal_tensor():
    """Test with a purely diagonal tensor."""
    print("\n" + "="*70)
    print("Test: Diagonal tensor")
    print("="*70)

    N = 3

    # Create diagonal tensor in charge basis (simplest case)
    arr_charge = np.zeros((N, N, N, N), dtype=complex)
    for q in range(N):
        arr_charge[q, q, q, q] = float(q + 1)  # 1, 2, 3

    print(f"Created diagonal tensor in charge basis")
    print(f"Elements: {[arr_charge[q,q,q,q] for q in range(N)]}")
    print(f"Trace(A @ A†) = {np.einsum('ijkl,ijkl->', arr_charge, arr_charge.conj())}")

    # Create TensorZ3
    A_z3 = TensorZ3.from_ndarray(arr_charge, shape=[[1,1,1]]*4, dirs=[1,1,-1,-1])

    print(f"\nTensorZ3 shape: {A_z3.shape}")
    print(f"TensorZ3 sects: {dict(A_z3.sects)}")

    # Get back
    arr_back = A_z3.to_ndarray()
    print(f"\nto_ndarray shape: {arr_back.shape}")
    print(f"Non-zero elements: {np.count_nonzero(np.abs(arr_back) > 1e-10)}")

    for q in range(N):
        print(f"  arr_back[{q},{q},{q},{q}] = {arr_back[q,q,q,q]}")

    print(f"Trace(back @ back†) = {np.einsum('ijkl,ijkl->', arr_back, arr_back.conj())}")

def test_z3_charge_conservation():
    """Test that TensorZ3 correctly handles charge conservation."""
    print("\n" + "="*70)
    print("Test: Z3 charge conservation")
    print("="*70)

    N = 3

    # Create a tensor that respects charge conservation: q0 + q1 = q2 + q3 mod 3
    arr_charge = np.zeros((N, N, N, N), dtype=complex)

    # Set all charge-conserving elements to 1
    count = 0
    for q0 in range(N):
        for q1 in range(N):
            for q2 in range(N):
                for q3 in range(N):
                    if (q0 + q1) % N == (q2 + q3) % N:
                        arr_charge[q0, q1, q2, q3] = 1.0
                        count += 1

    print(f"Created charge-conserving tensor with {count} non-zero elements")
    print(f"Trace(A @ A†) = {np.einsum('ijkl,ijkl->', arr_charge, arr_charge.conj())}")

    # Create TensorZ3
    A_z3 = TensorZ3.from_ndarray(arr_charge, shape=[[1,1,1]]*4, dirs=[1,1,-1,-1])

    print(f"\nTensorZ3 shape: {A_z3.shape}")
    print(f"Number of sectors: {len(A_z3.sects)}")

    # Count total elements
    total_z3_elements = sum(block.size for block in A_z3.sects.values())
    print(f"Total elements in TensorZ3 sectors: {total_z3_elements}")

    # Get back
    arr_back = A_z3.to_ndarray()
    print(f"\nto_ndarray shape: {arr_back.shape}")
    print(f"Non-zero in arr_back: {np.count_nonzero(np.abs(arr_back) > 1e-10)}")
    print(f"Trace(back @ back†) = {np.einsum('ijkl,ijkl->', arr_back, arr_back.conj())}")

    # Direct comparison
    diff = np.linalg.norm(arr_back - arr_charge) / np.linalg.norm(arr_charge)
    print(f"Relative difference: {diff:.2e}")

def test_manual_z3_construction():
    """Manually construct a TensorZ3 to understand the data layout."""
    print("\n" + "="*70)
    print("Test: Manual TensorZ3 construction")
    print("="*70)

    N = 3

    # Create TensorZ3 manually sector by sector
    # For 4-leg tensor with qhape [[1,1,1]]*4:
    # - Each leg has 3 sectors (q=0,1,2), each of dimension 1
    # - Valid sectors have q0 + q1 = q2 + q3 (mod 3) with appropriate dirs

    # With dirs = [1,1,-1,-1]:
    # Charge conservation: q0 * 1 + q1 * 1 + q2 * (-1) + q3 * (-1) = 0 mod 3
    # i.e., q0 + q1 = q2 + q3 mod 3

    # List all valid sectors
    valid_sects = []
    for q0 in range(N):
        for q1 in range(N):
            for q2 in range(N):
                for q3 in range(N):
                    if (q0 + q1) % N == (q2 + q3) % N:
                        valid_sects.append((q0, q1, q2, q3))

    print(f"Valid sectors: {len(valid_sects)}")

    # Create sects dict
    sects = {}
    for qvec in valid_sects:
        # Each sector is a 1x1x1x1 array
        sects[qvec] = np.array([[[[1.0 + 0j]]]])  # All ones for simplicity

    # Create TensorZ3 directly
    qhape = [[1, 1, 1]] * 4
    dirs = [1, 1, -1, -1]

    A_z3 = TensorZ3(shape=qhape, qhape=qhape, sects=sects, dirs=dirs, dtype=complex)

    print(f"Manually created TensorZ3:")
    print(f"  shape: {A_z3.shape}")
    print(f"  Number of sectors: {len(A_z3.sects)}")

    # Get as array
    arr = A_z3.to_ndarray()
    print(f"  to_ndarray shape: {arr.shape}")
    print(f"  Non-zero elements: {np.count_nonzero(np.abs(arr) > 1e-10)}")
    print(f"  Sum of elements: {np.sum(arr)}")
    print(f"  Trace(A @ A†) = {np.einsum('ijkl,ijkl->', arr, arr.conj())}")

    # Compare with expected
    arr_expected = np.zeros((N, N, N, N), dtype=complex)
    for qvec in valid_sects:
        arr_expected[qvec] = 1.0

    diff = np.linalg.norm(arr - arr_expected) / np.linalg.norm(arr_expected)
    print(f"  Difference from expected: {diff:.2e}")

def test_from_ndarray_internals():
    """Understand what from_ndarray does internally."""
    print("\n" + "="*70)
    print("Test: from_ndarray internals")
    print("="*70)

    N = 3

    # Simple charge-conserving tensor
    arr = np.zeros((N, N, N, N), dtype=complex)
    for q0 in range(N):
        for q1 in range(N):
            q_in = (q0 + q1) % N
            arr[q0, q1, q_in, 0] = 1.0  # q2 = q_in, q3 = 0 satisfies q0+q1 = q2+q3 when q_in = q0+q1

    # Wait, let me make all conserved elements = 1
    arr = np.zeros((N, N, N, N), dtype=complex)
    for q0 in range(N):
        for q1 in range(N):
            for q2 in range(N):
                q3 = (q0 + q1 - q2) % N
                arr[q0, q1, q2, q3] = 1.0

    print(f"Input array non-zeros: {np.count_nonzero(arr)}")

    # Try different dirs to see effect
    for dirs in [[1,1,1,1], [1,1,-1,-1], [1,-1,1,-1], [1,-1,-1,1]]:
        A_z3 = TensorZ3.from_ndarray(arr, shape=[[1,1,1]]*4, dirs=dirs)
        arr_back = A_z3.to_ndarray()
        n_nz = np.count_nonzero(np.abs(arr_back) > 1e-10)
        trace = np.einsum('ijkl,ijkl->', arr_back, arr_back.conj())
        print(f"dirs={dirs}: n_sectors={len(A_z3.sects)}, n_nonzero={n_nz}, trace={trace.real:.1f}")

if __name__ == "__main__":
    test_simple_tensor()
    test_diagonal_tensor()
    test_z3_charge_conservation()
    test_manual_z3_construction()
    test_from_ndarray_internals()
