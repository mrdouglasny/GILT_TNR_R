#!/usr/bin/env python3
"""
Investigate how TensorZ3.svd() works and why it gives different singular values.

Key insight from previous test: The charge-basis matrix is NOT block-diagonal
after the standard (0,2)×(1,3) leg reshaping used in TRG.

The question is: what DOES TensorZ3.svd() compute?
"""

import sys
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr/GiltTNR')
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr')

import numpy as np
from GiltTNR2D_Potts import get_initial_tensor_potts, get_initial_tensor_potts_z3
from tensors.symmetrytensors import TensorZ3
from tensors.tensor import Tensor

def investigate_charge_structure():
    """Understand the charge structure of the Potts tensor."""
    print("="*70)
    print("Understanding charge structure of Potts tensor")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))
    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})

    print(f"\nTensorZ3 shape: {A_z3.shape}")
    print(f"Number of sectors: {len(A_z3.sects)}")

    # The tensor has 4 legs, each with 3 charge sectors
    # But NOT all 3^4 = 81 charge combinations are present!
    # Only those satisfying charge conservation: q0 + q1 = q2 + q3 (mod 3)

    print("\nNon-zero sectors (q0, q1, q2, q3) where q0+q1 = q2+q3 mod 3:")
    for qvec, block in sorted(A_z3.sects.items()):
        q0, q1, q2, q3 = qvec
        if (q0 + q1) % 3 == (q2 + q3) % 3:
            print(f"  {qvec}: shape={block.shape}, |val|={np.abs(block.flat[0]):.6f}")
        else:
            print(f"  {qvec}: UNEXPECTED - charge not conserved!")

    # Count sectors per combined charge
    print("\nSectors grouped by (q0+q1) mod 3:")
    for total_charge in range(3):
        count = 0
        for qvec in A_z3.sects.keys():
            q0, q1, q2, q3 = qvec
            if (q0 + q1) % 3 == total_charge:
                count += 1
        print(f"  Total charge {total_charge}: {count} sectors")

def investigate_svd_reshaping():
    """Understand how TensorZ3 reshapes for SVD."""
    print("\n" + "="*70)
    print("Understanding TensorZ3 SVD reshaping")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))
    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    # TRG uses svd([0,2], [1,3]) to split the tensor
    # Let's see what happens with different leg groupings

    print("\n1. Standard TRG split: legs [0,2] vs [1,3]")
    print("   (This combines non-adjacent legs)")

    # For TensorZ3, this creates a matrix where:
    # Row index has combined charge (q0 + q2) mod 3
    # Col index has combined charge (q1 + q3) mod 3
    # For charge-conserving tensor: (q0+q1) = (q2+q3) implies (q0+q2) can differ from (q1+q3)!

    # Let's enumerate which row/col charges can couple
    print("\n   Checking which (q0+q2, q1+q3) pairs can be non-zero:")
    for qvec in A_z3.sects.keys():
        q0, q1, q2, q3 = qvec
        q_row = (q0 + q2) % 3
        q_col = (q1 + q3) % 3
        print(f"   {qvec}: row_charge={q_row}, col_charge={q_col}, delta={q_row - q_col}")

    # Count unique (q_row, q_col) pairs
    pairs = set()
    for qvec in A_z3.sects.keys():
        q0, q1, q2, q3 = qvec
        q_row = (q0 + q2) % 3
        q_col = (q1 + q3) % 3
        pairs.add((q_row, q_col))

    print(f"\n   Unique (row_charge, col_charge) pairs: {sorted(pairs)}")
    print(f"   Number of pairs: {len(pairs)}")
    print("   If all pairs have same q_row=q_col, matrix is block-diagonal")
    print("   Otherwise, matrix has off-diagonal blocks!")

    # Check if block-diagonal
    is_block_diag = all(p[0] == p[1] for p in pairs)
    print(f"\n   Matrix is block-diagonal: {is_block_diag}")

    print("\n2. Alternative split: legs [0,1] vs [2,3]")
    print("   (This combines adjacent legs - bra vs ket)")

    pairs2 = set()
    for qvec in A_z3.sects.keys():
        q0, q1, q2, q3 = qvec
        q_row = (q0 + q1) % 3
        q_col = (q2 + q3) % 3
        pairs2.add((q_row, q_col))

    print(f"   Unique (row_charge, col_charge) pairs: {sorted(pairs2)}")
    is_block_diag2 = all(p[0] == p[1] for p in pairs2)
    print(f"   Matrix is block-diagonal: {is_block_diag2}")

def compare_svd_groupings():
    """Compare SVD with different leg groupings."""
    print("\n" + "="*70)
    print("Comparing SVD with different leg groupings")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))
    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    arr_z3 = A_z3.to_ndarray()
    arr_plain = A_plain.to_ndarray()

    print("\n1. Grouping [0,2] vs [1,3] (TRG style):")

    # Plain tensor
    mat_trg_plain = arr_plain.transpose(0, 2, 1, 3).reshape(9, 9)
    _, S_plain_trg, _ = np.linalg.svd(mat_trg_plain)

    # Z3 tensor (convert to array then reshape)
    mat_trg_z3 = arr_z3.transpose(0, 2, 1, 3).reshape(9, 9)
    _, S_z3_trg, _ = np.linalg.svd(mat_trg_z3)

    print(f"   Plain: {np.sort(S_plain_trg)[::-1]}")
    print(f"   Z3:    {np.sort(S_z3_trg)[::-1]}")

    diff = np.linalg.norm(np.sort(S_plain_trg)[::-1] - np.sort(S_z3_trg)[::-1])
    print(f"   Difference: {diff:.2e}")

    print("\n2. Grouping [0,1] vs [2,3] (bra-ket style):")

    mat_bk_plain = arr_plain.reshape(9, 9)
    mat_bk_z3 = arr_z3.reshape(9, 9)

    _, S_plain_bk, _ = np.linalg.svd(mat_bk_plain)
    _, S_z3_bk, _ = np.linalg.svd(mat_bk_z3)

    print(f"   Plain: {np.sort(S_plain_bk)[::-1]}")
    print(f"   Z3:    {np.sort(S_z3_bk)[::-1]}")

    diff = np.linalg.norm(np.sort(S_plain_bk)[::-1] - np.sort(S_z3_bk)[::-1])
    print(f"   Difference: {diff:.2e}")

    print("\n3. TensorZ3.svd([0,2], [1,3]) - what it actually returns:")
    U, S, V = A_z3.svd([0, 2], [1, 3])
    S_arr = S.to_ndarray()
    print(f"   TensorZ3.svd S values: {np.sort(np.abs(S_arr.flatten()))[::-1]}")

    print("\n4. TensorZ3.svd([0,1], [2,3]):")
    U2, S2, V2 = A_z3.svd([0, 1], [2, 3])
    S2_arr = S2.to_ndarray()
    print(f"   TensorZ3.svd S values: {np.sort(np.abs(S2_arr.flatten()))[::-1]}")

def understand_tensorz3_svd_internals():
    """Look at TensorZ3 SVD implementation."""
    print("\n" + "="*70)
    print("Understanding TensorZ3 SVD internals")
    print("="*70)

    # Read the source to understand what it does
    import inspect
    from tensors.symmetrytensors import TensorZ3

    # Get the svd method
    print("\nTensorZ3.svd is inherited from abeliantensor.AbelianTensor")
    print("Key insight: It does SVD on each CHARGE SECTOR separately")
    print("This is correct for block-diagonal matrices")
    print("BUT: [0,2] vs [1,3] grouping creates NON-block-diagonal structure!")

    beta_c = np.log(1 + np.sqrt(3))
    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})

    # What [0,2] vs [1,3] does:
    # Combines legs 0 and 2 into a single "row" leg
    # Combines legs 1 and 3 into a single "col" leg
    #
    # For TensorZ3, each leg has charges 0,1,2
    # Combined leg has charges (q0+q2) mod 3
    #
    # The issue: charge conservation says q0+q1 = q2+q3 (mod 3)
    # This does NOT imply q0+q2 = q1+q3 (mod 3)!
    #
    # Example: (0,1,0,1) satisfies 0+1=0+1 (mod 3) ✓
    # But row charge = 0+0=0, col charge = 1+1=2 ≠ 0
    #
    # So the reshaped matrix has off-diagonal blocks!

    # The TensorZ3 SVD still does per-sector SVD
    # But now sectors are (row_charge, col_charge) pairs
    # Since row_charge ≠ col_charge for some sectors,
    # these become rectangular blocks, not square!

    print("\nExample showing why [0,2] vs [1,3] breaks block structure:")
    print("  Sector (0,1,0,1): charge conservation 0+1=0+1 ✓")
    print("    Row charge = (0+0) mod 3 = 0")
    print("    Col charge = (1+1) mod 3 = 2")
    print("    This sector goes in M[row_0, col_2] block = OFF-DIAGONAL!")

    print("\nContrast with [0,1] vs [2,3]:")
    print("  Sector (0,1,0,1): charge conservation 0+1=0+1 ✓")
    print("    Row charge = (0+1) mod 3 = 1")
    print("    Col charge = (0+1) mod 3 = 1")
    print("    This sector goes in M[row_1, col_1] block = DIAGONAL ✓")

def check_trg_leg_grouping():
    """Check what leg grouping the actual GiltTNR code uses."""
    print("\n" + "="*70)
    print("Checking actual TRG code leg grouping")
    print("="*70)

    # The TRG algorithm typically does:
    # A[l,r,u,d] -> split along (l,u) vs (r,d) or (l,d) vs (r,u)
    #
    # In the standard convention:
    # - legs 0,1 are "left/right" (horizontal)
    # - legs 2,3 are "up/down" (vertical)
    #
    # TRG step 1: SVD on (left,up) vs (right,down) = [0,2] vs [1,3]
    # TRG step 2: SVD on (left,down) vs (right,up) = [0,3] vs [1,2]

    # For Z3 symmetric tensors:
    # charge conservation: q_l + q_r = q_u + q_d (mod 3)
    # or equivalently: q_l + q_r + q_u + q_d = 0 (mod 3) with signed convention

    # With [0,2] vs [1,3]:
    # combined row charge = q_l + q_u (mod 3)
    # combined col charge = q_r + q_d (mod 3)
    # For charge conservation: q_l + q_r = q_u + q_d
    # This does NOT mean q_l + q_u = q_r + q_d

    print("\nThe issue: TRG uses [l,u] vs [r,d] leg grouping")
    print("For Z3 charge conservation: q_l + q_r = q_u + q_d (or sum=0)")
    print("But this doesn't imply (q_l + q_u) = (q_r + q_d)!")
    print("\nSo the reshaped matrix is NOT block-diagonal in charge space.")
    print("TensorZ3.svd() assumes block-diagonal structure → wrong singular values!")

if __name__ == "__main__":
    investigate_charge_structure()
    investigate_svd_reshaping()
    compare_svd_groupings()
    understand_tensorz3_svd_internals()
    check_trg_leg_grouping()
