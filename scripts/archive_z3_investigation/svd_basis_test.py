#!/usr/bin/env python3
"""
Test whether SVD singular values are basis-independent.

Key mathematical fact: The singular values of a matrix M are the square roots
of the eigenvalues of M†M (or MM†). This should be basis-independent.

BUT: For a block-diagonal matrix, doing SVD on each block separately gives
the same singular values as doing SVD on the full matrix (just with different
U and V matrices that preserve block structure).

Let's verify this and see why TensorZ3 gets different singular values.
"""

import numpy as np

def build_dft_matrix(N):
    """Build N×N DFT matrix: F[s,q] = ω^{sq}/√N where ω = e^{2πi/N}"""
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega**(s*q) for q in range(N)] for s in range(N)]) / np.sqrt(N)
    return F

def test_block_svd_equivalence():
    """Test that block SVD gives same singular values as full SVD."""
    print("="*70)
    print("Test 1: Block SVD vs Full SVD for block-diagonal matrix")
    print("="*70)

    N = 3
    block_size = 3

    # Create a block-diagonal matrix
    blocks = []
    for q in range(N):
        np.random.seed(42 + q)
        B = np.random.randn(block_size, block_size) + 0.1 * np.random.randn(block_size, block_size) * 1j
        blocks.append(B)

    # Full matrix (block diagonal)
    M = np.zeros((N*block_size, N*block_size), dtype=complex)
    for q, B in enumerate(blocks):
        i0 = q * block_size
        M[i0:i0+block_size, i0:i0+block_size] = B

    print(f"Matrix shape: {M.shape}")
    print(f"Block sizes: {block_size} x {block_size} each")

    # Full SVD
    _, S_full, _ = np.linalg.svd(M)
    S_full_sorted = np.sort(S_full)[::-1]

    # Block-wise SVD
    S_blocks = []
    for q, B in enumerate(blocks):
        _, S_block, _ = np.linalg.svd(B)
        S_blocks.extend(S_block)
    S_blocks_sorted = np.sort(np.array(S_blocks))[::-1]

    print(f"\nFull SVD singular values (sorted): {S_full_sorted[:10]}")
    print(f"Block SVD singular values (sorted): {S_blocks_sorted[:10]}")

    diff = np.linalg.norm(S_full_sorted - S_blocks_sorted) / np.linalg.norm(S_full_sorted)
    print(f"\nRelative difference: {diff:.2e}")
    print("✓ Block SVD = Full SVD for truly block-diagonal matrix" if diff < 1e-10 else "✗ DIFFERENT!")

def test_tensorz3_vs_plain_svd():
    """Test the actual Potts tensor SVD."""
    import sys
    sys.path.insert(0, '/workspaces/rg/ekrgilttrnr/GiltTNR')
    sys.path.insert(0, '/workspaces/rg/ekrgilttrnr')
    from GiltTNR2D_Potts import get_initial_tensor_potts, get_initial_tensor_potts_z3

    print("\n" + "="*70)
    print("Test 2: Potts tensor SVD - TensorZ3 vs Plain")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))
    N = 3

    # Get both representations
    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    arr_z3 = A_z3.to_ndarray()
    arr_plain = A_plain.to_ndarray()

    # Convert Z3 to spin basis for direct comparison
    F = build_dft_matrix(N)
    Fdag = F.conj().T
    arr_z3_spin = np.einsum('ai,bj,ijkl,kc,ld->abcd', Fdag, Fdag, arr_z3, F, F)

    print(f"arr_z3 (charge basis) shape: {arr_z3.shape}")
    print(f"arr_plain (spin basis) shape: {arr_plain.shape}")
    print(f"arr_z3_spin shape: {arr_z3_spin.shape}")

    # Check they're the same tensor in different bases
    diff = np.linalg.norm(arr_z3_spin - arr_plain) / np.linalg.norm(arr_plain)
    print(f"\nDifference between Z3→spin and plain: {diff:.2e}")

    # Now do SVD on both (as matrices, combining legs)
    mat_z3 = arr_z3.transpose(0, 2, 1, 3).reshape(9, 9)
    mat_plain = arr_plain.transpose(0, 2, 1, 3).reshape(9, 9)
    mat_z3_spin = arr_z3_spin.transpose(0, 2, 1, 3).reshape(9, 9)

    # SVD on plain tensor
    _, S_plain, _ = np.linalg.svd(mat_plain)

    # SVD on z3 in charge basis (full matrix SVD)
    _, S_z3_full, _ = np.linalg.svd(mat_z3)

    # SVD on z3 converted to spin basis
    _, S_z3_spin, _ = np.linalg.svd(mat_z3_spin)

    print(f"\nSingular values (full SVD):")
    print(f"  Plain (spin basis):  {np.sort(S_plain)[::-1]}")
    print(f"  Z3 (charge basis):   {np.sort(S_z3_full)[::-1]}")
    print(f"  Z3→spin:             {np.sort(S_z3_spin)[::-1]}")

    # Compare
    S_plain_sorted = np.sort(S_plain)[::-1]
    S_z3_sorted = np.sort(S_z3_full)[::-1]

    diff = np.linalg.norm(S_plain_sorted - S_z3_sorted) / np.linalg.norm(S_plain_sorted)
    print(f"\nPlain vs Z3-charge (full SVD): {diff:.2e}")

    # Now do BLOCK-wise SVD on charge basis (what TensorZ3 actually does)
    print("\n" + "-"*50)
    print("Block-wise SVD on charge basis matrix:")

    # The matrix mat_z3 should be block-diagonal in charge basis
    # Let's verify this and extract blocks

    # For a Z3 tensor, the matrix M[i,j] = A[a,b,c,d] with i=(a,c), j=(b,d)
    # has block structure based on charge

    # Check block structure
    print("\nChecking block-diagonal structure of mat_z3...")

    # In charge basis, row index i = a*3 + c has charge (q_a + q_c) mod 3
    # col index j = b*3 + d has charge (q_b + q_d) mod 3

    for q_row in range(3):
        for q_col in range(3):
            # Find indices with these charges
            row_idx = []
            col_idx = []
            for a in range(3):
                for c in range(3):
                    if (a + c) % 3 == q_row:
                        row_idx.append(a*3 + c)
            for b in range(3):
                for d in range(3):
                    if (b + d) % 3 == q_col:
                        col_idx.append(b*3 + d)

            block = mat_z3[np.ix_(row_idx, col_idx)]
            norm = np.linalg.norm(block)
            if q_row != q_col and norm > 1e-10:
                print(f"  Off-diagonal block ({q_row},{q_col}): norm = {norm:.2e} (should be ~0)")
            elif q_row == q_col:
                print(f"  Diagonal block ({q_row},{q_col}): norm = {norm:.2e}, shape = {block.shape}")

    # Now do per-block SVD
    S_per_block = []
    for q in range(3):
        row_idx = []
        col_idx = []
        for a in range(3):
            for c in range(3):
                if (a + c) % 3 == q:
                    row_idx.append(a*3 + c)
        for b in range(3):
            for d in range(3):
                if (b + d) % 3 == q:
                    col_idx.append(b*3 + d)

        block = mat_z3[np.ix_(row_idx, col_idx)]
        _, S_block, _ = np.linalg.svd(block)
        S_per_block.extend(S_block)
        print(f"\nBlock q={q}: singular values = {S_block}")

    S_per_block_sorted = np.sort(np.array(S_per_block))[::-1]

    print(f"\nAll per-block singular values (sorted): {S_per_block_sorted}")
    print(f"Full matrix SVD (sorted):               {S_z3_sorted}")

    diff = np.linalg.norm(S_per_block_sorted - S_z3_sorted) / np.linalg.norm(S_z3_sorted)
    print(f"\nPer-block vs Full SVD on same matrix: {diff:.2e}")

def test_what_tensorz3_actually_does():
    """Check what the TensorZ3 SVD actually does."""
    import sys
    sys.path.insert(0, '/workspaces/rg/ekrgilttrnr/GiltTNR')
    sys.path.insert(0, '/workspaces/rg/ekrgilttrnr')
    from GiltTNR2D_Potts import get_initial_tensor_potts_z3
    from tensors.symmetrytensors import TensorZ3

    print("\n" + "="*70)
    print("Test 3: What TensorZ3.svd() actually computes")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))
    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})

    print(f"TensorZ3 shape: {A_z3.shape}")
    print(f"Sectors: {A_z3.sects}")

    # Do SVD using TensorZ3's method
    # The tensor has 4 legs with Z3 charges
    # We want to split into (legs 0,1) and (legs 2,3)

    # TensorZ3 svd signature
    print("\nCalling A_z3.svd([0,1], [2,3])...")
    U, S, V = A_z3.svd([0,1], [2,3])

    print(f"U shape: {U.shape}")
    print(f"S shape: {S.shape}")
    print(f"V shape: {V.shape}")

    # S is a 1-leg TensorZ3 containing diagonal elements
    S_arr = S.to_ndarray()
    print(f"S as array: {S_arr}")
    print(f"S diagonal: {np.diag(S_arr) if len(S_arr.shape) == 2 else S_arr}")

if __name__ == "__main__":
    test_block_svd_equivalence()
    test_tensorz3_vs_plain_svd()
    test_what_tensorz3_actually_does()
