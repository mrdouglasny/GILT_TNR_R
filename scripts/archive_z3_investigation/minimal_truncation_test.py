#!/usr/bin/env python3
"""
Minimal test case for TensorZ3 vs plain tensor truncation.

Strategy:
1. Create a simple tensor that I fully understand
2. Apply truncation SVD using both TensorZ3 and plain methods
3. Compare results in the same basis
"""

import sys
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr/GiltTNR')
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr')

import numpy as np
from tensors.symmetrytensors import TensorZ3
from tensors.tensor import Tensor

def build_dft_matrix(N):
    """Build N×N DFT matrix: F[s,q] = ω^{sq}/√N where ω = e^{2πi/N}"""
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega**(s*q) for q in range(N)] for s in range(N)]) / np.sqrt(N)
    return F

def transform_charge_to_spin(arr_charge, N=3):
    """Transform tensor from charge basis to spin basis."""
    F = build_dft_matrix(N)
    Fdag = F.conj().T
    return np.einsum('ia,jb,abcd,ck,dl->ijkl', Fdag, Fdag, arr_charge, F, F)

def transform_spin_to_charge(arr_spin, N=3):
    """Transform tensor from spin basis to charge basis."""
    F = build_dft_matrix(N)
    Fdag = F.conj().T
    return np.einsum('ai,bj,ijkl,kc,ld->abcd', F, F, arr_spin, Fdag, Fdag)

def create_z3_symmetric_tensor(N=3, values=None):
    """
    Create a simple Z3 symmetric tensor in spin basis.

    The tensor will have A[s1,s2,s3,s4] ≠ 0 only when s1+s2 = s3+s4 mod 3.
    """
    if values is None:
        values = {0: 3.0, 1: 2.0, 2: 1.0}  # Value for each total charge

    arr = np.zeros((N, N, N, N), dtype=complex)
    for s1 in range(N):
        for s2 in range(N):
            for s3 in range(N):
                for s4 in range(N):
                    if (s1 + s2) % N == (s3 + s4) % N:
                        total_charge = (s1 + s2) % N
                        arr[s1, s2, s3, s4] = values[total_charge]
    return arr

def test_basic_svd():
    """Test that TensorZ3 and plain tensor give same SVD for a simple tensor."""
    print("="*70)
    print("Test 1: Basic SVD comparison")
    print("="*70)

    N = 3

    # Create a simple Z3 symmetric tensor in spin basis
    arr_spin = create_z3_symmetric_tensor(N)
    print(f"Created Z3-symmetric tensor in spin basis: shape {arr_spin.shape}")

    # Convert to charge basis
    arr_charge = transform_spin_to_charge(arr_spin, N)
    print(f"Converted to charge basis: shape {arr_charge.shape}")

    # Create TensorZ3 from charge basis array
    # TensorZ3 expects qhape = [[1]*N]*4 for 3D tensor with N sectors per leg
    qhape = [[1, 1, 1] for _ in range(4)]
    A_z3 = TensorZ3.from_ndarray(arr_charge, shape=qhape)
    print(f"TensorZ3 shape: {A_z3.shape}")

    # Create plain Tensor from spin basis
    A_plain = Tensor.from_ndarray(arr_spin)

    # Do SVD with legs [0,1] vs [2,3] - this gives block-diagonal structure
    print("\nSVD with legs [0,1] vs [2,3]:")

    # TensorZ3 SVD
    U_z3, S_z3, V_z3 = A_z3.svd([0, 1], [2, 3])
    S_z3_vals = np.sort(np.abs(S_z3.to_ndarray().flatten()))[::-1]
    print(f"  TensorZ3 S values: {S_z3_vals}")

    # Plain Tensor SVD (via reshape + numpy)
    mat_plain = arr_spin.reshape(N*N, N*N)
    _, S_plain, _ = np.linalg.svd(mat_plain)
    S_plain_vals = np.sort(S_plain)[::-1]
    print(f"  Plain S values: {S_plain_vals}")

    diff = np.linalg.norm(S_z3_vals - S_plain_vals) / np.linalg.norm(S_plain_vals)
    print(f"  Relative difference: {diff:.2e}")

    # Also test [0,2] vs [1,3] - this is what TRG uses
    print("\nSVD with legs [0,2] vs [1,3]:")

    U_z3_2, S_z3_2, V_z3_2 = A_z3.svd([0, 2], [1, 3])
    S_z3_2_vals = np.sort(np.abs(S_z3_2.to_ndarray().flatten()))[::-1]
    print(f"  TensorZ3 S values: {S_z3_2_vals}")

    mat_plain_2 = arr_spin.transpose(0, 2, 1, 3).reshape(N*N, N*N)
    _, S_plain_2, _ = np.linalg.svd(mat_plain_2)
    S_plain_2_vals = np.sort(S_plain_2)[::-1]
    print(f"  Plain S values: {S_plain_2_vals}")

    diff_2 = np.linalg.norm(S_z3_2_vals - S_plain_2_vals) / np.linalg.norm(S_plain_2_vals)
    print(f"  Relative difference: {diff_2:.2e}")

def test_truncated_svd():
    """Test truncated SVD."""
    print("\n" + "="*70)
    print("Test 2: Truncated SVD comparison")
    print("="*70)

    N = 3

    # Create tensor with more structure
    values = {0: 5.0, 1: 3.0, 2: 1.0}
    arr_spin = create_z3_symmetric_tensor(N, values)

    # Add some variation within each sector
    for s1 in range(N):
        for s2 in range(N):
            for s3 in range(N):
                for s4 in range(N):
                    if (s1 + s2) % N == (s3 + s4) % N:
                        arr_spin[s1, s2, s3, s4] *= (1 + 0.1 * (s1 + s2 + s3 + s4))

    print(f"Created tensor with variation within sectors")

    # Full SVD
    mat = arr_spin.transpose(0, 2, 1, 3).reshape(9, 9)
    U_full, S_full, Vh_full = np.linalg.svd(mat)

    print(f"\nFull singular values: {S_full}")

    # Truncate to chi = 6
    chi = 6
    print(f"\nTruncating to chi={chi}:")

    # Plain truncation: keep top chi singular values
    U_trunc = U_full[:, :chi]
    S_trunc = S_full[:chi]
    Vh_trunc = Vh_full[:chi, :]

    # Reconstruct
    mat_reconst = U_trunc @ np.diag(S_trunc) @ Vh_trunc
    arr_reconst = mat_reconst.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3)

    print(f"  Plain truncation error: {np.linalg.norm(mat - mat_reconst):.2e}")

    # TensorZ3 truncation - need to understand what it does
    arr_charge = transform_spin_to_charge(arr_spin, N)
    A_z3 = TensorZ3.from_ndarray(arr_charge, shape=[[1,1,1]]*4)

    # Try truncated SVD using TensorZ3 with chi limit
    # The svd method returns full SVD; truncation happens elsewhere
    U_z3, S_z3, V_z3 = A_z3.svd([0, 2], [1, 3])

    print(f"\n  TensorZ3 S shape: {S_z3.shape}")
    print(f"  TensorZ3 S values: {np.sort(np.abs(S_z3.to_ndarray().flatten()))[::-1]}")

    # The actual truncation in TRG is done by the abeliantensor's svd method
    # with a chis parameter. Let me try that.

def test_tensor_contraction():
    """Test that tensor contractions give same result in both bases."""
    print("\n" + "="*70)
    print("Test 3: Tensor contraction comparison")
    print("="*70)

    N = 3

    # Create two simple tensors
    arr_spin_A = create_z3_symmetric_tensor(N, {0: 3.0, 1: 2.0, 2: 1.0})
    arr_spin_B = create_z3_symmetric_tensor(N, {0: 2.0, 1: 1.0, 2: 0.5})

    # Contract leg 1 of A with leg 0 of B
    # Result: C[i,k,l] = Σ_j A[i,j,k,l] * B[j,...]
    # Wait, both are 4-leg tensors. Let me contract A[i,j,k,l] with B[j,m,n,o]

    # Actually, let's do a simpler test: trace(A @ A^T)
    trace_spin = np.einsum('ijkl,ijkl->', arr_spin_A, arr_spin_A.conj())
    print(f"Trace(A @ A†) in spin basis: {trace_spin:.6f}")

    # Same in charge basis
    arr_charge_A = transform_spin_to_charge(arr_spin_A, N)
    trace_charge = np.einsum('ijkl,ijkl->', arr_charge_A, arr_charge_A.conj())
    print(f"Trace(A @ A†) in charge basis: {trace_charge:.6f}")

    # These should match due to unitarity of DFT
    print(f"Difference: {abs(trace_spin - trace_charge):.2e}")

    # Now test TensorZ3 trace
    A_z3 = TensorZ3.from_ndarray(arr_charge_A, shape=[[1,1,1]]*4)

    # Compute trace as contraction
    trace_z3 = np.einsum('ijkl,ijkl->', A_z3.to_ndarray(), A_z3.to_ndarray().conj())
    print(f"Trace(A @ A†) via TensorZ3: {trace_z3:.6f}")

def test_trg_contraction_step():
    """Test a single TRG-like contraction."""
    print("\n" + "="*70)
    print("Test 4: TRG-like contraction")
    print("="*70)

    N = 3

    # Create a simple tensor
    values = {0: 5.0, 1: 3.0, 2: 1.0}
    arr_spin = create_z3_symmetric_tensor(N, values)

    # TRG contracts A with itself by:
    # 1. SVD on A to get A = U @ sqrt(S) @ sqrt(S) @ V
    # 2. Contract to form new tensor

    # Let's just do the SVD part
    mat = arr_spin.transpose(0, 2, 1, 3).reshape(9, 9)
    U, S, Vh = np.linalg.svd(mat)

    # Form half-tensors
    sqrtS = np.sqrt(S)
    left = (U @ np.diag(sqrtS)).reshape(3, 3, 9)   # [s0, s2, new]
    right = (np.diag(sqrtS) @ Vh).reshape(9, 3, 3)  # [new, s1, s3]

    # Contract to form new tensor (TRG step)
    # new_A[new_l, new_r, new_u, new_d] = left[l,u,new_l] * left[r,d,new_r] * ...
    # This is getting complex. Let me simplify.

    # Just verify: if we reconstruct, do we get back the original?
    mat_reconst = U @ np.diag(S) @ Vh
    diff = np.linalg.norm(mat - mat_reconst) / np.linalg.norm(mat)
    print(f"SVD reconstruction error: {diff:.2e}")

    # Now do same with TensorZ3
    arr_charge = transform_spin_to_charge(arr_spin, N)
    A_z3 = TensorZ3.from_ndarray(arr_charge, shape=[[1,1,1]]*4)

    U_z3, S_z3, V_z3 = A_z3.svd([0, 2], [1, 3])

    # Reconstruct
    A_z3_reconst = U_z3 @ S_z3 @ V_z3

    arr_reconst = A_z3_reconst.to_ndarray()
    arr_original = A_z3.to_ndarray()

    diff_z3 = np.linalg.norm(arr_reconst - arr_original) / np.linalg.norm(arr_original)
    print(f"TensorZ3 SVD reconstruction error: {diff_z3:.2e}")

if __name__ == "__main__":
    test_basic_svd()
    test_truncated_svd()
    test_tensor_contraction()
    test_trg_contraction_step()
