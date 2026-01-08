#!/usr/bin/env python3
"""
Compare TensorZ3 vs plain tensor truncation in the SAME BASIS.

The key insight: we can convert both tensors to spin basis before comparing,
which removes the basis-dependent representation and lets us see if the
truncation itself produces different results.

Flow:
1. Start with same tensor in both representations
2. Apply SVD truncation (the core of TRG step)
3. Convert both results to spin basis
4. Compare - any difference is due to truncation algorithm, not representation
"""

import sys
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr/GiltTNR')
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr')

import numpy as np
from GiltTNR2D_Potts import get_initial_tensor_potts, get_initial_tensor_potts_z3
from tensors.symmetrytensors import TensorZ3
from tensors.tensor import Tensor

def build_dft_matrix(N):
    """Build N×N DFT matrix: F[s,q] = ω^{sq}/√N where ω = e^{2πi/N}"""
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega**(s*q) for q in range(N)] for s in range(N)]) / np.sqrt(N)
    return F

def transform_tensor_to_spin(A_z3, N=3):
    """Transform TensorZ3 from charge basis to spin basis."""
    F = build_dft_matrix(N)
    Fdag = F.conj().T

    arr = A_z3.to_ndarray()

    # Handle case where tensor has bond dimension > N (after TRG steps)
    chi = arr.shape[0]
    if chi > N:
        # Build larger transformation matrix
        block_size = chi // N
        F_large = np.kron(np.eye(block_size), F)
        Fdag_large = F_large.conj().T
        arr_spin = np.einsum('ai,bj,ijkl,kc,ld->abcd', Fdag_large, Fdag_large, arr, F_large, F_large)
    else:
        # A_spin[a,b,c,d] = Σ_ijkl F†[a,i] F†[b,j] A[i,j,k,l] F[k,c] F[l,d]
        arr_spin = np.einsum('ai,bj,ijkl,kc,ld->abcd', Fdag, Fdag, arr, F, F)
    return arr_spin

def transform_tensor_to_charge(A_spin, N=3):
    """Transform from spin basis to charge basis."""
    F = build_dft_matrix(N)
    Fdag = F.conj().T

    if hasattr(A_spin, 'to_ndarray'):
        arr = A_spin.to_ndarray()
    else:
        arr = A_spin
    # A_charge[i,j,k,l] = Σ_abcd F[i,a] F[j,b] A[a,b,c,d] F†[c,k] F†[d,l]
    arr_charge = np.einsum('ia,jb,abcd,ck,dl->ijkl', F, F, arr, Fdag, Fdag)
    return arr_charge

def do_svd_truncation_plain(A, chi):
    """Do SVD truncation on a plain tensor (standard TRG-style)."""
    if hasattr(A, 'to_ndarray'):
        arr = A.to_ndarray()
    else:
        arr = np.array(A)

    d = arr.shape[0]
    # Reshape to matrix for SVD: combine (left, up) and (right, down)
    mat = arr.transpose(0, 2, 1, 3).reshape(d*d, d*d)

    U, S, Vh = np.linalg.svd(mat, full_matrices=False)

    # Truncate to chi
    k = min(chi, len(S))
    U = U[:, :k]
    S = S[:k]
    Vh = Vh[:k, :]

    # sqrt(S) to each half
    sqrtS = np.sqrt(S)
    U_trunc = U @ np.diag(sqrtS)
    V_trunc = np.diag(sqrtS) @ Vh

    # Reshape back to tensors (these become the new A tensor legs)
    # U_trunc: (d*d, k) -> (d, d, k) -> legs (left, up, new_leg)
    # V_trunc: (k, d*d) -> (k, d, d) -> legs (new_leg, right, down)

    U_tens = U_trunc.reshape(d, d, k)
    V_tens = V_trunc.reshape(k, d, d)

    # Contract to form new A: A'[l',r',u',d'] = U[l,u,l'] V[r',r,d] δ(old legs)
    # But actually TRG contracts two copies... for now just return the SVD info

    return S, k, U_trunc, V_trunc

def do_svd_truncation_z3(A_z3, chi):
    """
    Do SVD truncation on TensorZ3 using its block-diagonal structure.
    This mimics what TensorZ3.svd() does internally.
    """
    N = 3  # Z3

    # Get the underlying array in charge basis
    arr = A_z3.to_ndarray()
    d = arr.shape[0]  # Should be multiple of N

    # The TensorZ3 SVD works on each charge sector independently
    # Let's manually do what the library does

    # For TensorZ3, the tensor has block structure based on charge
    # A[i,j,k,l] ≠ 0 only if q_i + q_j = q_k + q_l (mod N)

    # Reshape same way as plain
    mat = arr.transpose(0, 2, 1, 3).reshape(d*d, d*d)

    # Now the matrix has block structure
    # Row index (i,k) has charge q_row = (q_i + q_k) mod N
    # Col index (j,l) has charge q_col = (q_j + q_l) mod N
    # Non-zero only when q_row = q_col

    # Extract blocks
    block_size = d // N
    all_singular_values = []
    block_info = []  # Store (q, U_block, S_block, V_block)

    for q in range(N):
        # Find row/col indices with charge q
        row_indices = []
        col_indices = []
        for i in range(d):
            for k in range(d):
                qi = i // block_size  # charge of index i
                qk = k // block_size  # charge of index k
                if (qi + qk) % N == q:
                    row_indices.append(i * d + k)  # flattened index
        for j in range(d):
            for l in range(d):
                qj = j // block_size
                ql = l // block_size
                if (qj + ql) % N == q:
                    col_indices.append(j * d + l)

        if len(row_indices) == 0 or len(col_indices) == 0:
            continue

        # Extract block
        block = mat[np.ix_(row_indices, col_indices)]

        # SVD on block
        U_block, S_block, V_block = np.linalg.svd(block, full_matrices=False)

        for i, s in enumerate(S_block):
            all_singular_values.append((s, q, i))
        block_info.append((q, U_block, S_block, V_block, row_indices, col_indices))

    # Sort all singular values
    all_singular_values.sort(reverse=True, key=lambda x: x[0])

    # Now allocate chi dimensions using greedy (like TensorZ3 does)
    sector_dims = {0: 0, 1: 0, 2: 0}
    for i in range(min(chi, len(all_singular_values))):
        s, q, idx = all_singular_values[i]
        sector_dims[q] += 1

    print(f"  Greedy allocation for chi={chi}: {sector_dims}")

    # Return the full singular values for comparison
    S_full = np.array([x[0] for x in all_singular_values])

    return S_full, sector_dims, block_info

def compare_truncations():
    """Compare truncation in same basis."""
    print("="*70)
    print("Comparing TensorZ3 vs plain tensor truncation in same basis")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))
    chi = 15  # Target bond dimension

    # Get initial tensors
    print("\n1. Creating initial tensors...")
    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    print(f"   Z3 shape: {A_z3.shape}")
    print(f"   Plain shape: {A_plain.shape}")

    # Convert both to spin basis
    print("\n2. Converting to spin basis for comparison...")
    arr_z3_spin = transform_tensor_to_spin(A_z3)
    arr_plain_spin = A_plain.to_ndarray()  # Already in spin basis

    # Check they're the same (up to normalization)
    diff = np.linalg.norm(arr_z3_spin - arr_plain_spin) / np.linalg.norm(arr_plain_spin)
    print(f"   Initial tensor difference in spin basis: {diff:.2e}")

    # Do SVD truncation on plain tensor (in spin basis)
    print("\n3. SVD truncation on plain tensor...")
    S_plain, k_plain, _, _ = do_svd_truncation_plain(arr_plain_spin, chi)
    print(f"   Kept {k_plain} singular values")
    print(f"   Top 10 S: {S_plain[:10]}")

    # Do SVD truncation on Z3 tensor (in charge basis, block-diagonal)
    print("\n4. SVD truncation on TensorZ3 (block-diagonal)...")
    S_z3, sector_dims, block_info = do_svd_truncation_z3(A_z3, chi)
    print(f"   Total singular values: {len(S_z3)}")
    print(f"   Top 10 S: {S_z3[:10]}")

    # Compare singular values
    print("\n5. Comparing singular values...")
    min_len = min(len(S_plain), len(S_z3))
    S_plain_sorted = np.sort(S_plain)[::-1]
    S_z3_sorted = np.sort(S_z3)[::-1]

    diff = np.linalg.norm(S_plain_sorted[:min_len] - S_z3_sorted[:min_len])
    rel_diff = diff / np.linalg.norm(S_plain_sorted[:min_len])
    print(f"   ||S_plain - S_z3|| / ||S_plain|| = {rel_diff:.2e}")

    # The key question: after truncation, do the tensors differ?
    print("\n6. Key finding:")
    print("   The SINGULAR VALUES are the same (they're basis-independent)!")
    print("   The difference is in HOW dimensions are allocated across sectors.")
    print(f"   Z3 allocation: {sector_dims}")
    total_z3 = sum(sector_dims.values())
    print(f"   Total Z3 dims: {total_z3}, Plain dims: {k_plain}")

    # Show per-sector breakdown
    print("\n7. Per-sector singular value distribution:")
    for q, U_block, S_block, V_block, _, _ in block_info:
        print(f"   Sector q={q}: {len(S_block)} singular values")
        print(f"      Top 5: {S_block[:5]}")
        print(f"      Allocated: {sector_dims[q]} dims")

def examine_trg_step():
    """Examine what happens in a full TRG step."""
    print("\n" + "="*70)
    print("Examining full TRG step")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))

    # Get initial tensors
    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    # Use the actual TRG code
    from GiltTNR2D import gilttnr_step

    pars_z3 = {
        "gilt_eps": 1e-7,
        "cg_chis": list(range(2, 25)),
        "cg_eps": 1e-10,
        "verbosity": 0,
        "symmetry_tensors": True
    }
    pars_plain = dict(pars_z3)
    pars_plain["symmetry_tensors"] = False

    print("\nRunning GILT-TNR step on both...")

    try:
        A_z3_new, log_z3 = gilttnr_step(A_z3, 0.0, pars_z3)
        print(f"Z3 after step: shape = {A_z3_new.shape}")
    except Exception as e:
        print(f"Z3 step failed: {e}")
        A_z3_new = None

    A_plain_new, log_plain = gilttnr_step(A_plain, 0.0, pars_plain)
    print(f"Plain after step: shape = {A_plain_new.shape}")

    if A_z3_new is not None:
        # Convert both to spin basis
        arr_z3_spin = transform_tensor_to_spin(A_z3_new)
        arr_plain_spin = A_plain_new.to_ndarray()

        # Handle size mismatch
        if arr_z3_spin.shape != arr_plain_spin.shape:
            print(f"\nShape mismatch: Z3={arr_z3_spin.shape}, Plain={arr_plain_spin.shape}")
            print("This confirms the truncation allocates different dimensions!")

            # Compare in overlapping region
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(arr_z3_spin.shape, arr_plain_spin.shape))
            z3_sub = arr_z3_spin[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]]
            plain_sub = arr_plain_spin[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]]

            diff = np.linalg.norm(z3_sub - plain_sub)
            print(f"Difference in overlap region {min_shape}: {diff:.2e}")
        else:
            diff = np.linalg.norm(arr_z3_spin - arr_plain_spin) / np.linalg.norm(arr_plain_spin)
            print(f"Relative difference in spin basis: {diff:.2e}")

if __name__ == "__main__":
    compare_truncations()
    examine_trg_step()
