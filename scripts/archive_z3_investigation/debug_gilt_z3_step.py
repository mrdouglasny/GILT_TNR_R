#!/usr/bin/env python3
"""
Debug exactly where Z₃ charge conservation breaks during Gilt-TNR step.
Compare Z2 (Ising) vs Z3 (Potts) behavior at each sub-operation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from tensors import Tensor, TensorZ2, TensorZ3

def check_z2_charge(arr, name):
    """Check Z₂ charge conservation for Ising tensor."""
    non_zero = 0
    charge_conserved = 0
    for idx in np.ndindex(*arr.shape):
        if np.abs(arr[idx]) > 1e-10:
            non_zero += 1
            # For dirs [1,1,-1,-1], Z₂ conservation: sum(d*q) mod 2 = 0
            dirs = [1, 1, -1, -1]
            charge_sum = sum(d * q for d, q in zip(dirs, idx))
            if charge_sum % 2 == 0:
                charge_conserved += 1
    pct = 100 * charge_conserved / non_zero if non_zero > 0 else 0
    print(f"  {name}: {charge_conserved}/{non_zero} = {pct:.1f}% charge-conserving")
    return pct

def check_z3_charge(arr, name):
    """Check Z₃ charge conservation for Potts tensor."""
    non_zero = 0
    charge_conserved = 0
    for idx in np.ndindex(*arr.shape):
        if np.abs(arr[idx]) > 1e-10:
            non_zero += 1
            # For dirs [1,1,-1,-1], Z₃ conservation: sum(d*q) mod 3 = 0
            dirs = [1, 1, -1, -1]
            charge_sum = sum(d * q for d, q in zip(dirs, idx))
            if charge_sum % 3 == 0:
                charge_conserved += 1
    pct = 100 * charge_conserved / non_zero if non_zero > 0 else 0
    print(f"  {name}: {charge_conserved}/{non_zero} = {pct:.1f}% charge-conserving")
    return pct

def test_svd_charge_conservation():
    """Test that SVD preserves charge sectors correctly."""
    print("=" * 70)
    print("Testing SVD charge conservation")
    print("=" * 70)

    # Test Z2
    print("\n--- Z2 SVD test ---")
    beta = np.log(1 + np.sqrt(2)) / 2
    W2 = np.array([[np.exp(beta), 1], [1, np.exp(beta)]])
    T2 = np.einsum('ab,bc,cd,da->abcd', W2, W2, W2, W2)

    # Hadamard transform
    u2 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    T2_charge = ncon((T2, u2, u2, u2.T.conj(), u2.T.conj()),
                     ([1,2,3,4], [-1,1], [-2,2], [3,-3], [4,-4]))

    check_z2_charge(T2_charge, "Z2 initial")

    # Create TensorZ2 and do SVD
    A2 = TensorZ2.from_ndarray(T2_charge, shape=[[1,1]]*4, qhape=[[0,1]]*4, dirs=[1,1,-1,-1])
    result = A2.svd([0,1], [2,3])
    U2, S2, V2 = result[0], result[1], result[2]

    # Contract back
    A2_recon = ncon((U2, S2.diag(), V2), ([-1,-2,1], [1,2], [2,-3,-4]))
    arr2_recon = A2_recon.to_ndarray()
    check_z2_charge(arr2_recon, "Z2 after SVD")

    print(f"  SVD reconstruction error: {np.max(np.abs(T2_charge - arr2_recon)):.2e}")

    # Test Z3
    print("\n--- Z3 SVD test ---")
    beta3 = np.log(1 + np.sqrt(3))
    W3 = np.ones((3, 3))
    np.fill_diagonal(W3, np.exp(beta3))
    T3 = np.einsum('ab,bc,cd,da->abcd', W3, W3, W3, W3)

    # Z3 DFT transform
    omega = np.exp(2j * np.pi / 3)
    F = np.array([[1, 1, 1], [1, omega, omega**2], [1, omega**2, omega]]) / np.sqrt(3)
    T3_charge = ncon((T3, F, F, F.T.conj(), F.T.conj()),
                     ([1,2,3,4], [-1,1], [-2,2], [3,-3], [4,-4]))
    T3_charge[np.abs(T3_charge) < 1e-10] = 0

    check_z3_charge(np.real(T3_charge), "Z3 initial")

    # Create TensorZ3 and do SVD
    A3 = TensorZ3.from_ndarray(T3_charge, shape=[[1,1,1]]*4, qhape=[[0,1,2]]*4,
                               dirs=[1,1,-1,-1], charge=0, invar=True)

    print(f"  TensorZ3 type: {type(A3)}")
    print(f"  TensorZ3 qodulus: {A3.qodulus}")
    print(f"  TensorZ3 shape: {A3.shape}")
    print(f"  TensorZ3 qhape: {A3.qhape}")
    print(f"  TensorZ3 dirs: {A3.dirs}")
    print(f"  TensorZ3 charge: {A3.charge}")
    print(f"  TensorZ3 invar: {A3.invar}")

    # Check sectors
    print(f"  TensorZ3 sectors: {list(A3.sects.keys())}")

    try:
        result = A3.svd([0,1], [2,3])
        U3, S3, V3 = result[0], result[1], result[2]
        print(f"  U3 shape: {U3.shape}, qhape: {U3.qhape}")
        print(f"  S3 shape: {S3.shape}, qhape: {S3.qhape}")
        print(f"  V3 shape: {V3.shape}, qhape: {V3.qhape}")

        # Contract back
        A3_recon = ncon((U3, S3.diag(), V3), ([-1,-2,1], [1,2], [2,-3,-4]))
        arr3_recon = A3_recon.to_ndarray()
        check_z3_charge(np.real(arr3_recon), "Z3 after SVD")

        print(f"  SVD reconstruction error: {np.max(np.abs(T3_charge - arr3_recon)):.2e}")
    except Exception as e:
        print(f"  ERROR in Z3 SVD: {e}")
        import traceback
        traceback.print_exc()

def test_ncon_charge_conservation():
    """Test that ncon preserves charge sectors."""
    print("\n" + "=" * 70)
    print("Testing ncon charge conservation")
    print("=" * 70)

    # Test Z2 contraction
    print("\n--- Z2 ncon test ---")
    beta = np.log(1 + np.sqrt(2)) / 2
    W2 = np.array([[np.exp(beta), 1], [1, np.exp(beta)]])
    T2 = np.einsum('ab,bc,cd,da->abcd', W2, W2, W2, W2)
    u2 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    T2_charge = ncon((T2, u2, u2, u2.T.conj(), u2.T.conj()),
                     ([1,2,3,4], [-1,1], [-2,2], [3,-3], [4,-4]))

    A2 = TensorZ2.from_ndarray(T2_charge, shape=[[1,1]]*4, qhape=[[0,1]]*4, dirs=[1,1,-1,-1])

    # Contract two copies (like building transfer matrix)
    T_row_2 = ncon((A2, A2), [[3,-101,4,-1], [4,-102,3,-2]])
    arr_row_2 = T_row_2.to_ndarray()

    print(f"  A2 shape: {A2.shape}")
    print(f"  T_row shape: {T_row_2.shape}")

    # Check charge conservation of result (different dirs for contracted tensor)
    non_zero = 0
    charge_conserved = 0
    # Result has dirs from uncontracted indices
    for idx in np.ndindex(*arr_row_2.shape):
        if np.abs(arr_row_2[idx]) > 1e-10:
            non_zero += 1
            # Assuming result has dirs [1,-1,1,-1] from original construction
            charge_sum = idx[0] - idx[1] + idx[2] - idx[3]
            if charge_sum % 2 == 0:
                charge_conserved += 1
    pct = 100 * charge_conserved / non_zero if non_zero > 0 else 0
    print(f"  T_row Z2: {charge_conserved}/{non_zero} = {pct:.1f}% charge-conserving")

    # Test Z3 contraction
    print("\n--- Z3 ncon test ---")
    beta3 = np.log(1 + np.sqrt(3))
    W3 = np.ones((3, 3))
    np.fill_diagonal(W3, np.exp(beta3))
    T3 = np.einsum('ab,bc,cd,da->abcd', W3, W3, W3, W3)
    omega = np.exp(2j * np.pi / 3)
    F = np.array([[1, 1, 1], [1, omega, omega**2], [1, omega**2, omega]]) / np.sqrt(3)
    T3_charge = ncon((T3, F, F, F.T.conj(), F.T.conj()),
                     ([1,2,3,4], [-1,1], [-2,2], [3,-3], [4,-4]))
    T3_charge[np.abs(T3_charge) < 1e-10] = 0

    A3 = TensorZ3.from_ndarray(T3_charge, shape=[[1,1,1]]*4, qhape=[[0,1,2]]*4,
                               dirs=[1,1,-1,-1], charge=0, invar=True)

    # Contract two copies
    T_row_3 = ncon((A3, A3), [[3,-101,4,-1], [4,-102,3,-2]])
    arr_row_3 = T_row_3.to_ndarray()

    print(f"  A3 shape: {A3.shape}")
    print(f"  T_row shape: {T_row_3.shape}")

    non_zero = 0
    charge_conserved = 0
    for idx in np.ndindex(*arr_row_3.shape):
        if np.abs(arr_row_3[idx]) > 1e-10:
            non_zero += 1
            charge_sum = idx[0] - idx[1] + idx[2] - idx[3]
            if charge_sum % 3 == 0:
                charge_conserved += 1
    pct = 100 * charge_conserved / non_zero if non_zero > 0 else 0
    print(f"  T_row Z3: {charge_conserved}/{non_zero} = {pct:.1f}% charge-conserving")

def test_gilt_operations():
    """Test individual Gilt operations for charge conservation."""
    print("\n" + "=" * 70)
    print("Testing Gilt sub-operations")
    print("=" * 70)

    # Import Gilt functions
    from GiltTNR2D import gilttnr_step
    from GiltTNR2D_Potts import get_initial_tensor_potts_relT
    from GiltTNR2D_Ising_benchmarks import get_initial_tensor

    chi = 12

    # Test Ising
    print("\n--- Ising Z2 through Gilt-TNR ---")
    pars_z2 = {
        'beta': np.log(1 + np.sqrt(2)) / 2,
        'gilt_eps': 1e-6,
        'cg_chis': [chi],
        'cg_eps': 1e-10,
        'verbosity': 0,
        'symmetry_tensors': True,
    }
    A_z2 = get_initial_tensor(pars_z2)
    print(f"  Initial type: {type(A_z2).__name__}")
    arr_z2 = A_z2.to_ndarray()
    check_z2_charge(arr_z2, "Z2 initial")

    A_z2_new, _ = gilttnr_step(A_z2, 0.0, pars_z2)
    print(f"  After step type: {type(A_z2_new).__name__}")
    arr_z2_new = A_z2_new.to_ndarray()
    check_z2_charge(arr_z2_new, "Z2 after Gilt-TNR")

    # Test Potts
    print("\n--- Potts Z3 through Gilt-TNR ---")
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
    print(f"  Initial type: {type(A_z3).__name__}")
    print(f"  Initial qodulus: {A_z3.qodulus if hasattr(A_z3, 'qodulus') else 'N/A'}")
    arr_z3 = A_z3.to_ndarray()
    check_z3_charge(np.real(arr_z3), "Z3 initial")

    A_z3_new, _ = gilttnr_step(A_z3, 0.0, pars_z3)
    print(f"  After step type: {type(A_z3_new).__name__}")
    print(f"  After step qodulus: {A_z3_new.qodulus if hasattr(A_z3_new, 'qodulus') else 'N/A'}")
    arr_z3_new = A_z3_new.to_ndarray()
    check_z3_charge(np.real(arr_z3_new), "Z3 after Gilt-TNR")

    # Compare shapes
    print("\n--- Shape comparison ---")
    print(f"  Z2 initial shape: {arr_z2.shape}")
    print(f"  Z2 after shape:   {arr_z2_new.shape}")
    print(f"  Z3 initial shape: {arr_z3.shape}")
    print(f"  Z3 after shape:   {arr_z3_new.shape}")

if __name__ == "__main__":
    test_svd_charge_conservation()
    test_ncon_charge_conservation()
    test_gilt_operations()
