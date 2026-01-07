#!/usr/bin/env python3
"""
Debug script to compare plain Tensor vs TensorZ3 for 3-state Potts.

Tests:
1. Partition function (trace) should match
2. Initial scaling dimensions should match
3. After one Gilt step, check if they diverge
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from tensors import Tensor, TensorZ3

def build_potts_tensor_plain(beta):
    """Plain Tensor version with vertex construction (like Ising)."""
    q = 3
    # Boltzmann weight matrix
    W = np.ones((q, q))
    np.fill_diagonal(W, np.exp(beta))

    # VERTEX construction: T[a,b,c,d] = W[a,b] W[b,c] W[c,d] W[d,a]
    T = np.einsum('ab,bc,cd,da->abcd', W, W, W, W)
    return Tensor.from_ndarray(T)


def build_potts_tensor_plain_svd(beta):
    """Plain Tensor version with SVD construction."""
    q = 3
    W = np.ones((q, q))
    np.fill_diagonal(W, np.exp(beta))
    eigenvalues, U = np.linalg.eigh(W)
    sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))
    A = U @ np.diag(sqrt_eigenvalues)
    T = np.einsum('sl,sr,su,sd->lrud', A, A, A, A)
    return Tensor.from_ndarray(T)

def build_potts_tensor_z3(beta):
    """TensorZ3 version with vertex construction (like Ising) + DFT transformation."""
    q = 3
    # Boltzmann weight matrix
    W = np.ones((q, q))
    np.fill_diagonal(W, np.exp(beta))

    # VERTEX construction (same as Ising): T[a,b,c,d] = W[a,b] W[b,c] W[c,d] W[d,a]
    T_spin = np.einsum('ab,bc,cd,da->abcd', W, W, W, W)

    # Z₃ DFT
    omega = np.exp(2j * np.pi / 3)
    F = np.array([
        [1, 1, 1],
        [1, omega, omega**2],
        [1, omega**2, omega]
    ]) / np.sqrt(3)
    F_dg = F.T.conj()

    # Transform: T'[k1,k2,k3,k4] = sum F[k1,s1] F[k2,s2] F†[s3,k3] F†[s4,k4] T[s1,s2,s3,s4]
    T_charge = ncon(
        (T_spin, F, F, F_dg, F_dg),
        ([1, 2, 3, 4], [-1, 1], [-2, 2], [3, -3], [4, -4])
    )

    # Clean up small values
    T_charge[np.abs(T_charge) < 1e-10] = 0

    dim = [1, 1, 1]
    qim = [0, 1, 2]
    dirs = [1, 1, -1, -1]

    return TensorZ3.from_ndarray(
        T_charge,
        shape=[dim]*4,
        qhape=[qim]*4,
        dirs=dirs,
        charge=0,
        invar=True
    )


def build_potts_tensor_z3_svd(beta):
    """TensorZ3 version with SVD construction + DFT transformation."""
    q = 3
    W = np.ones((q, q))
    np.fill_diagonal(W, np.exp(beta))
    eigenvalues, U = np.linalg.eigh(W)
    sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))
    A = U @ np.diag(sqrt_eigenvalues)
    T_spin = np.einsum('sl,sr,su,sd->lrud', A, A, A, A)

    # Z₃ DFT
    omega = np.exp(2j * np.pi / 3)
    F = np.array([
        [1, 1, 1],
        [1, omega, omega**2],
        [1, omega**2, omega]
    ]) / np.sqrt(3)
    F_dg = F.T.conj()

    # Transform: T'[k1,k2,k3,k4] = sum F[k1,s1] F[k2,s2] F†[s3,k3] F†[s4,k4] T[s1,s2,s3,s4]
    T_charge = ncon(
        (T_spin, F, F, F_dg, F_dg),
        ([1, 2, 3, 4], [-1, 1], [-2, 2], [3, -3], [4, -4])
    )

    # Clean up small values
    T_charge[np.abs(T_charge) < 1e-10] = 0

    dim = [1, 1, 1]
    qim = [0, 1, 2]
    dirs = [1, 1, -1, -1]

    return TensorZ3.from_ndarray(
        T_charge,
        shape=[dim]*4,
        qhape=[qim]*4,
        dirs=dirs,
        charge=0,
        invar=True
    )

def compute_trace(T):
    """Compute partition function trace: Tr(T) = sum_{ijkl} T[i,j,k,l] delta_{ij} delta_{kl}"""
    # Contract: T[i,i,k,k]
    return ncon((T,), [[1, 1, 2, 2]])

def compute_scaldims(A):
    """Extract scaling dimensions from transfer matrix."""
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

def main():
    beta_c = np.log(1 + np.sqrt(3))
    print(f"3-state Potts at β_c = {beta_c:.6f}")
    print("=" * 60)

    # Test 1: Initial tensors
    print("\n1. Building initial tensors...")
    T_plain = build_potts_tensor_plain(beta_c)
    T_z3 = build_potts_tensor_z3(beta_c)

    print(f"   Plain Tensor type: {type(T_plain)}")
    print(f"   TensorZ3 type: {type(T_z3)}")

    # Test 2: Compare traces (partition function contribution)
    print("\n2. Comparing traces (partition function)...")
    trace_plain = compute_trace(T_plain)
    trace_z3_obj = compute_trace(T_z3)
    # TensorZ3 trace returns a scalar tensor; extract value
    if hasattr(trace_z3_obj, 'defval'):
        trace_z3 = trace_z3_obj.defval
    else:
        trace_z3 = complex(trace_z3_obj)
    trace_z3 = np.real(trace_z3)  # Should be real
    print(f"   Plain Tensor trace: {trace_plain}")
    print(f"   TensorZ3 trace:     {trace_z3}")
    print(f"   Relative diff:      {abs(trace_plain - trace_z3) / abs(trace_plain):.2e}")

    # Test 3: Compare scaling dimensions
    print("\n3. Comparing initial scaling dimensions...")
    sd_plain = compute_scaldims(T_plain)
    sd_z3 = compute_scaldims(T_z3)
    print(f"   Plain: x_1={sd_plain[1]:.4f}, x_2={sd_plain[2]:.4f}, x_3={sd_plain[3]:.4f}")
    print(f"   Z3:    x_1={sd_z3[1]:.4f}, x_2={sd_z3[2]:.4f}, x_3={sd_z3[3]:.4f}")

    # Test 4: Check raw arrays
    print("\n4. Checking tensor arrays...")
    arr_plain = T_plain.to_ndarray()
    arr_z3 = T_z3.to_ndarray()

    print(f"   Plain shape: {arr_plain.shape}, dtype: {arr_plain.dtype}")
    print(f"   Z3 shape:    {arr_z3.shape}, dtype: {arr_z3.dtype}")
    print(f"   Plain max:   {np.max(np.abs(arr_plain)):.6f}")
    print(f"   Z3 max:      {np.max(np.abs(arr_z3)):.6f}")

    # The spin basis tensor should be related to charge basis by DFT
    # Check: T_charge should have non-zero elements only where charge conservation holds
    print("\n5. Checking charge conservation in Z3 tensor...")
    arr_z3_raw = T_z3.to_ndarray()
    non_zero_count = 0
    charge_conserved_count = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    val = arr_z3_raw[i, j, k, l]
                    if np.abs(val) > 1e-10:
                        non_zero_count += 1
                        # Charge conservation: i + j = k + l (mod 3) for dirs [1,1,-1,-1]
                        if (i + j) % 3 == (k + l) % 3:
                            charge_conserved_count += 1
                        else:
                            print(f"   Non-conserved: [{i},{j},{k},{l}] = {val:.6f}, "
                                  f"charges: {i}+{j}={(i+j)%3} vs {k}+{l}={(k+l)%3}")
    print(f"   Non-zero elements: {non_zero_count}")
    print(f"   Charge-conserved:  {charge_conserved_count}")

    # Test 5a-extra: Check what TensorZ3 stores
    print("\n5a-extra: TensorZ3 internal structure...")
    print(f"   T_z3.shape = {T_z3.shape}")
    print(f"   T_z3.qhape = {T_z3.qhape}")
    print(f"   T_z3.dirs = {T_z3.dirs}")
    print(f"   T_z3.charge = {T_z3.charge}")

    # Check a few blocks
    print(f"   Sample T_z3 blocks:")
    for key in [(0,0,0,0), (0,0,1,2), (1,1,1,1)]:
        if key in T_z3.sects:
            val = T_z3.sects[key]
            print(f"     {key}: {val.flatten()[0]:.4f}")

    # Compare to raw input (vertex construction)
    print("\n5a-extra2: Compare input vs stored (vertex construction)...")
    arr_z3_stored = T_z3.to_ndarray()

    # Build the input we passed to from_ndarray (VERTEX construction)
    q = 3
    W = np.ones((q, q))
    np.fill_diagonal(W, np.exp(beta_c))
    T_spin_check = np.einsum('ab,bc,cd,da->abcd', W, W, W, W)  # vertex construction

    omega = np.exp(2j * np.pi / 3)
    F = np.array([
        [1, 1, 1],
        [1, omega, omega**2],
        [1, omega**2, omega]
    ]) / np.sqrt(3)
    F_dg = F.T.conj()

    T_charge_input = ncon(
        (T_spin_check, F, F, F_dg, F_dg),
        ([1, 2, 3, 4], [-1, 1], [-2, 2], [3, -3], [4, -4])
    )
    T_charge_input[np.abs(T_charge_input) < 1e-10] = 0

    print(f"   Input[0,0,0,0]:  {T_charge_input[0,0,0,0]:.4f}")
    print(f"   Stored[0,0,0,0]: {arr_z3_stored[0,0,0,0]:.4f}")
    print(f"   Input[1,1,1,1]:  {T_charge_input[1,1,1,1]:.4f}")
    print(f"   Stored[1,1,1,1]: {arr_z3_stored[1,1,1,1]:.4f}")
    print(f"   Max diff:        {np.max(np.abs(T_charge_input - arr_z3_stored)):.4f}")

    # Test 5b: Check if basis transformation is reversible
    print("\n5b. Testing basis transformation reversibility...")

    # Get raw arrays
    arr_plain = T_plain.to_ndarray()

    # Apply F and F† to plain tensor
    omega = np.exp(2j * np.pi / 3)
    F = np.array([
        [1, 1, 1],
        [1, omega, omega**2],
        [1, omega**2, omega]
    ]) / np.sqrt(3)
    F_dg = F.T.conj()

    # Transform plain -> charge basis
    T_transformed = ncon(
        (arr_plain, F, F, F_dg, F_dg),
        ([1, 2, 3, 4], [-1, 1], [-2, 2], [3, -3], [4, -4])
    )

    # Transform back: charge -> spin basis
    # Inverse: F† on first two legs, F on last two
    T_back = ncon(
        (T_transformed, F_dg, F_dg, F, F),
        ([1, 2, 3, 4], [-1, 1], [-2, 2], [3, -3], [4, -4])
    )

    # Check roundtrip
    diff = np.max(np.abs(arr_plain - T_back))
    print(f"   Roundtrip error (should be ~0): {diff:.2e}")

    # The transformed tensor should match what we put in TensorZ3
    arr_z3 = T_z3.to_ndarray()
    diff_transform = np.max(np.abs(T_transformed - arr_z3))
    print(f"   Transform matches Z3: {diff_transform:.2e}")

    # Test 5c: Build TM from raw arrays to bypass TensorZ3 contraction
    print("\n5c. Building TM from raw arrays...")

    # TM from raw plain array
    arr_tm_plain_raw = np.einsum('abcd,cdeg->abeg', arr_plain, arr_plain).reshape(9, 9)
    eigs_plain_raw = np.abs(np.linalg.eigvals(arr_tm_plain_raw))
    eigs_plain_raw = -np.sort(-eigs_plain_raw)
    print(f"   Plain (raw) TM eigs: {eigs_plain_raw[:4]}")

    # TM from raw Z3 array (in charge basis)
    arr_tm_z3_raw = np.einsum('abcd,cdeg->abeg', arr_z3, arr_z3).reshape(9, 9)
    eigs_z3_raw = np.abs(np.linalg.eigvals(arr_tm_z3_raw))
    eigs_z3_raw = -np.sort(-eigs_z3_raw)
    print(f"   Z3 (raw) TM eigs:    {eigs_z3_raw[:4]}")

    # TM from transformed array (should match plain after transform)
    arr_tm_trans_raw = np.einsum('abcd,cdeg->abeg', T_transformed, T_transformed).reshape(9, 9)
    eigs_trans_raw = np.abs(np.linalg.eigvals(arr_tm_trans_raw))
    eigs_trans_raw = -np.sort(-eigs_trans_raw)
    print(f"   Transformed TM eigs: {eigs_trans_raw[:4]}")

    # Test 5d: The key insight - transformed TM should match plain TM
    print("\n5d. Key check: do basis-transformed TMs give same spectrum?")
    print(f"   Plain TM largest eig:       {eigs_plain_raw[0]:.6f}")
    print(f"   Charge-basis TM largest:    {eigs_trans_raw[0]:.6f}")
    if np.abs(eigs_plain_raw[0] - eigs_trans_raw[0]) < 1e-6:
        print("   ✓ Eigenvalues match - basis change preserves TM spectrum")
    else:
        print("   ✗ Eigenvalues DIFFER - something wrong with transformation")

    # Test 5e: Debug TensorZ3 ncon
    print("\n5e. TensorZ3 ncon vs raw contraction...")
    # Compare Z3 raw eigenvalues to transformed eigenvalues
    print(f"   Z3 raw TM eigs:       {eigs_z3_raw[:4]}")
    print(f"   Transformed TM eigs:  {eigs_trans_raw[:4]}")
    if np.allclose(sorted(eigs_z3_raw[:4]), sorted(eigs_trans_raw[:4]), rtol=1e-6):
        print("   ✓ Z3 raw and transformed match")
    else:
        print("   ✗ Z3 raw and transformed DIFFER")


    # Test 6: One Gilt step
    print("\n6. Running one Gilt-TNR step...")
    try:
        from GiltTNR2D import gilttnr_step

        pars = {
            'gilt_eps': 1e-6,
            'cg_chis': [12],
            'cg_eps': 1e-10,
            'verbosity': 0,
        }

        A1_plain, _ = gilttnr_step(T_plain, None, pars)
        A1_z3, _ = gilttnr_step(T_z3, None, pars)

        sd1_plain = compute_scaldims(A1_plain)
        sd1_z3 = compute_scaldims(A1_z3)

        print(f"   After 1 step:")
        print(f"   Plain: x_1={sd1_plain[1]:.4f}, x_2={sd1_plain[2]:.4f}, x_3={sd1_plain[3]:.4f}")
        print(f"   Z3:    x_1={sd1_z3[1]:.4f}, x_2={sd1_z3[2]:.4f}, x_3={sd1_z3[3]:.4f}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("Expected CFT: x_σ = 2/15 ≈ 0.1333, x_ε = 4/5 = 0.8")

if __name__ == "__main__":
    main()
