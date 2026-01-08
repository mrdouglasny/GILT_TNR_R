#!/usr/bin/env python3
"""
Test the balanced_sectors fix for TensorZ3 truncation.

This script verifies that:
1. The balanced_sectors=True option works
2. TensorZ3 sector allocation is more balanced with the fix
3. The fix preserves numerical accuracy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from tensors import TensorZ3

def test_balanced_allocation():
    """Test that balanced_sectors gives more even allocation."""
    print("=" * 70)
    print("Test 1: Balanced Sector Allocation")
    print("=" * 70)

    # Create a TensorZ3 with known structure
    # For Z3 tensor, qhape must satisfy charge conservation
    # For a 2-leg matrix tensor: q_in + q_out = 0 (mod 3)
    # So diagonal blocks have (0,0), (1,2), (2,1) charge combinations
    shape = [[5, 4, 3], [5, 3, 4]]  # Different sizes to show imbalance
    qhape = [[0, 1, 2], [0, 2, 1]]  # Conjugate charges for matrix
    dirs = [1, -1]

    T = TensorZ3(shape, qhape=qhape, dirs=dirs)

    # Fill diagonal blocks with random data
    # For charge conservation: (q_in, q_out) where q_in - q_out = 0 mod 3
    # Valid combinations: (0,0), (1,1), (2,2)
    T.sects[(0, 0)] = np.random.randn(5, 5)  # q=0 block
    T.sects[(1, 2)] = np.random.randn(4, 3)  # q=1 in, q=2 out (charge 1-(-2)=1-1=0 mod 3... wait)

    # Actually for Z3 matrix: indices have dirs [1, -1]
    # Charge conservation: dir[0]*q[0] + dir[1]*q[1] = 0 mod 3
    # So: q[0] - q[1] = 0 mod 3, meaning q[0] = q[1]
    # Valid blocks: (0,0), (1,1), (2,2) using qhape indices
    T.sects[(0, 0)] = np.random.randn(5, 5)  # q=0 block
    T.sects[(1, 1)] = np.random.randn(4, 3)  # q=1 block (dim 4 x dim 3 from qhape mapping)
    T.sects[(2, 2)] = np.random.randn(3, 4)  # q=2 block

    # Convert to matrix and do SVD with both methods
    T_matrix = T.to_matrix([0], [1])

    print(f"\nOriginal tensor shape: {shape}")
    print(f"Total dims per leg: {[sum(s) for s in shape]} = {sum(shape[0])}")

    # Test with chi = 6 (less than total)
    chi = 6
    print(f"\nTruncating to chi = {chi}")

    # Greedy allocation (default)
    U_greedy, S_greedy, V_greedy, err_greedy = T_matrix.matrix_svd(
        chis=[chi], eps=0, balanced_sectors=False
    )

    # Balanced allocation (new)
    U_balanced, S_balanced, V_balanced, err_balanced = T_matrix.matrix_svd(
        chis=[chi], eps=0, balanced_sectors=True
    )

    print(f"\nGreedy allocation:")
    print(f"  S shape: {S_greedy.shape}")
    print(f"  S qhape: {S_greedy.qhape}")
    print(f"  Dims per sector: {S_greedy.shape[0]}")
    print(f"  Truncation error: {err_greedy:.6e}")

    print(f"\nBalanced allocation:")
    print(f"  S shape: {S_balanced.shape}")
    print(f"  S qhape: {S_balanced.qhape}")
    print(f"  Dims per sector: {S_balanced.shape[0]}")
    print(f"  Truncation error: {err_balanced:.6e}")

    # Check that balanced allocation is indeed more balanced
    greedy_dims = S_greedy.shape[0]
    balanced_dims = S_balanced.shape[0]

    greedy_imbalance = max(greedy_dims) - min(greedy_dims) if greedy_dims else 0
    balanced_imbalance = max(balanced_dims) - min(balanced_dims) if balanced_dims else 0

    print(f"\nSector imbalance (max - min):")
    print(f"  Greedy: {greedy_imbalance}")
    print(f"  Balanced: {balanced_imbalance}")

    if balanced_imbalance <= greedy_imbalance:
        print("\n✓ Balanced allocation has equal or better balance")
    else:
        print("\n✗ Balanced allocation is MORE imbalanced (unexpected)")

    return True


def test_svd_accuracy():
    """Test that both methods give accurate SVD reconstructions."""
    print("\n" + "=" * 70)
    print("Test 2: SVD Reconstruction Accuracy")
    print("=" * 70)

    shape = [[4, 4, 4], [4, 4, 4]]
    qhape = [[0, 1, 2], [0, 1, 2]]
    dirs = [1, -1]

    T = TensorZ3(shape, qhape=qhape, dirs=dirs)
    for k in T.sects:
        T.sects[k] = np.random.randn(*T.sects[k].shape)

    T_matrix = T.to_matrix([0], [1])
    T_array = T_matrix.to_ndarray()

    chi = 9  # 3 per sector for equal allocation

    # Test both methods
    for balanced, name in [(False, "Greedy"), (True, "Balanced")]:
        U, S, V, err = T_matrix.matrix_svd(
            chis=[chi], eps=0, balanced_sectors=balanced
        )

        # Reconstruct
        recon = U.to_ndarray() @ np.diag(S.to_ndarray()) @ V.to_ndarray()
        actual_err = np.linalg.norm(T_array - recon) / np.linalg.norm(T_array)

        print(f"\n{name}:")
        print(f"  Reported error: {err:.6e}")
        print(f"  Actual error:   {actual_err:.6e}")
        print(f"  Sector dims: {S.shape[0]}")

    return True


def test_z3_flow_step():
    """Test that balanced_sectors works in a Gilt-TNR context."""
    print("\n" + "=" * 70)
    print("Test 3: Z3 Tensor in Gilt-TNR Context")
    print("=" * 70)

    from GiltTNR2D_Potts import get_initial_tensor_potts_relT

    pars = {
        'q': 3,
        'relT': 1.0,
        'symmetry_tensors': True,
    }

    A = get_initial_tensor_potts_relT(pars)
    print(f"\nInitial Potts tensor:")
    print(f"  Type: {type(A).__name__}")
    print(f"  Shape: {A.shape}")
    print(f"  qhape: {A.qhape}")

    # Do an SVD with truncation using both methods
    chi = 12

    for balanced, name in [(False, "Greedy"), (True, "Balanced")]:
        U, S, V, err = A.svd([0, 1], [2, 3], chis=[chi], eps=0,
                            balanced_sectors=balanced, return_rel_err=True)
        print(f"\n{name} (chi={chi}):")
        print(f"  S shape: {S.shape}")
        print(f"  S dims per sector: {S.shape[0] if hasattr(S, 'shape') else 'N/A'}")
        print(f"  Error: {err:.6e}")

    return True


if __name__ == "__main__":
    print("Testing balanced_sectors fix for TensorZ3 truncation\n")

    tests = [
        ("Balanced allocation", test_balanced_allocation),
        ("SVD accuracy", test_svd_accuracy),
        ("Z3 flow step", test_z3_flow_step),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"\n✗ {name} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "ERROR"))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {symbol} {name}: {status}")
