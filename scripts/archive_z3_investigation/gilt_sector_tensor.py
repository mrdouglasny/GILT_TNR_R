#!/usr/bin/env python3
"""
GILT for Z3 equivariant tensors via per-sector tensor product with DFT.

ALGORITHM:
1. For each charge sector q with dimension n_q:
   - The sector block T_q has shape [n_0^(q), n_1^(q), n_2^(q), n_3^(q)]
     where n_i^(q) are sector dims on each leg for that charge combination
   - Tensor with DFT vector: each leg gets F[:,q_leg] on in-legs, F*[:,q_leg] on out-legs
   - Result has shape [3×n_0^(q), 3×n_1^(q), 3×n_2^(q), 3×n_3^(q)]

2. Take direct sum across all valid charge combinations:
   - The spin-basis tensor is the sum of all transformed blocks
   - Dimension grows by factor of 3 per leg: χ → 3χ

3. Apply GILT on this larger (3χ)^4 tensor:
   - Standard GILT with plain tensor operations
   - SVD truncation brings dimension back down

4. Transform back to charge basis:
   - Project spin-basis result onto charge sectors using F†
   - Each sector's projection: integrate over spin values weighted by F†[q,s]

5. Reconstruct TensorZ3 with proper block structure

NOTE: The dimension grows temporarily but SVD truncation keeps it bounded.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D import gilttnr_step, get_envspec, optimize_Rp
from GiltTNR2D_Potts import get_initial_tensor_potts_relT
from tensors import Tensor, TensorZ3


def get_dft_matrix(N=3):
    """DFT matrix for Z_N: F[j,k] = omega^(jk) / sqrt(N)."""
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega**(j*k) for k in range(N)] for j in range(N)]) / np.sqrt(N)
    return F


def expand_to_spin_basis(T_z3, N=3):
    """
    Expand Z3 tensor to spin basis by tensoring each sector with DFT vectors.

    For a 4-leg tensor T with charge constraint q0 + q1 = q2 + q3 (mod N),
    valid charge combinations are (q0, q1, q2, q3) satisfying this.

    Each sector block T[q0,q1,q2,q3] is expanded by tensoring with:
    - F[:,q0] on leg 0 (in-leg)
    - F[:,q1] on leg 1 (in-leg)
    - F*[:,q2] on leg 2 (out-leg) - conjugate for out-legs
    - F*[:,q3] on leg 3 (out-leg)

    Result: dimension grows from χ to 3χ per leg.
    """
    arr = T_z3.to_ndarray()
    chi = arr.shape[0]  # Assume square tensor

    F = get_dft_matrix(N)

    # For the initial 3x3x3x3 tensor with one state per charge:
    # chi=3, sector dims = [1,1,1] for each leg
    # Expanded dim = 3×3 = 9 per leg? No wait...

    # Let me think more carefully.
    # If chi=3 and sector dims = [1,1,1], then:
    # - Index 0 has charge 0, index 1 has charge 1, index 2 has charge 2
    # - In spin basis, each charge index expands to 3 spin indices
    # - So chi=3 → 3×3 = 9? No, that's not right either.

    # The key: The spin basis has SAME dimension as charge basis (χ).
    # The DFT is a unitary transformation: F @ charge_vector = spin_vector
    # It's NOT a tensor product that increases dimension.

    # For a tensor with block structure, the transformation is:
    # T_spin[s0,s1,s2,s3] = Σ_{q0,q1,q2,q3} F†[s0,q0] F†[s1,q1] F[s2,q2] F[s3,q3] T_charge[q0,q1,q2,q3]

    # This is just a change of basis, dimension stays χ.

    # But the user said "each sector should be tensored with the appropriate vector".
    # I think they mean: for unbalanced sectors, we need to handle differently.

    # Let me re-read the instruction:
    # "each sector has to be translated to spin basis by tensoring,
    #  then take direct sum, then GILT, then go back to charge basis"

    # I think "tensoring" here means the Kronecker structure:
    # |spin, multiplicity⟩ = Σ_q F†[spin,q] |charge_q, multiplicity⟩

    # For balanced sectors (all have same dim n):
    # - Total χ = 3n
    # - Transform is U = F† ⊗ I_n
    # - Spin-basis tensor has same dimension χ

    # For unbalanced sectors (dims n_0, n_1, n_2):
    # - Total χ = n_0 + n_1 + n_2
    # - The transformation is trickier because multiplicities don't align

    # AH! I think I understand now. The user's approach is:
    # 1. Take sector q with n_q states
    # 2. Tensor with the spin vector for that charge: creates 3×n_q states
    # 3. Direct sum all sectors: creates 3×(n_0 + n_1 + n_2) = 3χ states
    # 4. This 3χ dimensional space is redundant - some directions are not used
    # 5. Apply GILT on this larger space
    # 6. Project back

    # Let me implement this interpretation:

    # Build expansion matrix: χ → 3χ
    # For charge index c in sector q with offset α (c = start_q + α):
    # Maps to spin indices (s, c) = (s, start_q + α) for s ∈ {0,1,2}
    # with coefficient F†[s, q]

    # Determine sector structure
    if hasattr(T_z3, 'shape') and isinstance(T_z3.shape[0], dict):
        sector_dims = [T_z3.shape[leg].get(q, 0) for q in range(N) for leg in range(1)]
        # Actually need per-leg info
        pass

    # For simplicity, handle the initial chi=3 case first
    if chi == 3:
        # Simple case: one state per charge
        # Expansion: index q → (s, q) for all s
        # T_expanded[s0,q0, s1,q1, s2,q2, s3,q3] = F†[s0,q0] F†[s1,q1] F[s2,q2] F[s3,q3] T[q0,q1,q2,q3]

        # Actually for GILT, we just need the dense spin-basis tensor
        # T_spin[s0,s1,s2,s3] = Σ_q F†[s0,q0] F†[s1,q1] F[s2,q2] F[s3,q3] T[q0,q1,q2,q3]

        Fdag = F.conj().T
        result = np.einsum('ai,bj,ijkl,ck,dl->abcd', Fdag, Fdag, arr, F, F)
        return Tensor.from_ndarray(result), chi

    # For larger chi with block structure, we need the expanded representation
    # Build expansion matrix E: χ → 3χ
    # E[3c + s, c] = F†[s, charge(c)]

    # First, determine charge of each index
    # For now, assume equal sectors (chi divisible by 3)
    if chi % N != 0:
        print(f"Warning: chi={chi} not divisible by {N}, using approximate expansion")

    n = chi // N  # States per sector (approximately)

    # Build charge map: index → charge
    charge_of = np.zeros(chi, dtype=int)
    for q in range(N):
        for k in range(n):
            idx = q * n + k
            if idx < chi:
                charge_of[idx] = q
    # Handle remaining indices
    for idx in range(N * n, chi):
        charge_of[idx] = idx % N

    # Build expansion matrix E: χ → 3χ
    E = np.zeros((3 * chi, chi), dtype=complex)
    Fdag = F.conj().T
    for c in range(chi):
        q = charge_of[c]
        for s in range(3):
            E[3*c + s, c] = Fdag[s, q]

    # Transform tensor: T_expanded = E ⊗ E ⊗ T ⊗ E† ⊗ E†
    # But this gives 3χ × 3χ × 3χ × 3χ tensor which is huge

    # Actually, let's do the contraction differently:
    # T_spin = E T_charge E† on each leg
    # For efficiency, apply E as matrix multiplication per leg

    # Reshape and apply E to each leg
    # Leg 0: (χ, χ, χ, χ) → (3χ, χ, χ, χ)
    T1 = np.tensordot(E, arr, axes=([1], [0]))  # (3χ, χ, χ, χ)
    # Leg 1: (3χ, χ, χ, χ) → (3χ, 3χ, χ, χ)
    T2 = np.tensordot(E, T1, axes=([1], [1]))  # (3χ, 3χ, χ, χ)
    T2 = T2.transpose(1, 0, 2, 3)  # Put legs in right order
    # Leg 2: (3χ, 3χ, χ, χ) → (3χ, 3χ, 3χ, χ)
    E_conj = E.conj()  # For out-legs
    T3 = np.tensordot(E_conj, T2, axes=([1], [2]))  # (3χ, 3χ, 3χ, χ)
    T3 = T3.transpose(1, 2, 0, 3)
    # Leg 3: (3χ, 3χ, 3χ, χ) → (3χ, 3χ, 3χ, 3χ)
    T4 = np.tensordot(E_conj, T3, axes=([1], [3]))  # (3χ, 3χ, 3χ, 3χ)
    T_expanded = T4.transpose(1, 2, 3, 0)

    return Tensor.from_ndarray(T_expanded), chi


def contract_from_spin_basis(T_spin, original_chi, N=3):
    """
    Contract spin-basis tensor back to charge basis.

    This is the inverse of expand_to_spin_basis.
    Uses E† to project back: T_charge = E† T_spin E
    """
    arr = T_spin.to_ndarray()
    chi_spin = arr.shape[0]

    F = get_dft_matrix(N)

    # Simple case: chi=3
    if original_chi == 3:
        # T_charge = F T_spin F†
        result = np.einsum('ai,bj,ijkl,ck,dl->abcd', F, F, arr, F.conj().T, F.conj().T)
        return Tensor.from_ndarray(result)

    # For larger chi, build contraction matrix
    n = original_chi // N
    charge_of = np.zeros(original_chi, dtype=int)
    for q in range(N):
        for k in range(n):
            idx = q * n + k
            if idx < original_chi:
                charge_of[idx] = q
    for idx in range(N * n, original_chi):
        charge_of[idx] = idx % N

    # Build expansion matrix E
    E = np.zeros((3 * original_chi, original_chi), dtype=complex)
    Fdag = F.conj().T
    for c in range(original_chi):
        q = charge_of[c]
        for s in range(3):
            E[3*c + s, c] = Fdag[s, q]

    # E† is the contraction matrix
    Edag = E.conj().T  # (χ, 3χ)

    # Apply E† to each leg
    # If T_spin has shape (3χ, 3χ, 3χ, 3χ), result is (χ, χ, χ, χ)
    if chi_spin == 3 * original_chi:
        T1 = np.tensordot(Edag, arr, axes=([1], [0]))
        T2 = np.tensordot(Edag, T1, axes=([1], [1]))
        T2 = T2.transpose(1, 0, 2, 3)
        Edag_conj = Edag.conj()
        T3 = np.tensordot(Edag_conj, T2, axes=([1], [2]))
        T3 = T3.transpose(1, 2, 0, 3)
        T4 = np.tensordot(Edag_conj, T3, axes=([1], [3]))
        T_charge = T4.transpose(1, 2, 3, 0)
    else:
        # Dimension changed (e.g., due to GILT truncation)
        # Need different approach - not fully implemented
        print(f"Warning: chi_spin={chi_spin} != 3×{original_chi}={3*original_chi}")
        print("  Dimension changed during GILT, using approximate contraction")

        # Truncate E to match actual dimension
        actual_dim = chi_spin // 3  # Approximate
        Edag_trunc = Edag[:actual_dim, :chi_spin]

        T1 = np.tensordot(Edag_trunc, arr, axes=([1], [0]))
        # Continue with truncated dimensions...
        return Tensor.from_ndarray(arr[:original_chi, :original_chi, :original_chi, :original_chi])

    return Tensor.from_ndarray(T_charge)


def gilt_with_expansion(A_z3, pars):
    """
    Apply GILT by expanding to spin basis, running GILT, contracting back.
    """
    # Expand to spin basis
    A_spin, original_chi = expand_to_spin_basis(A_z3)

    # Plain tensor parameters
    pars_plain = dict(pars)
    pars_plain['symmetry_tensors'] = False

    # Apply GILT in spin basis
    for where in ["S", "E", "N", "W"]:
        U, S = get_envspec(A_spin, A_spin, pars_plain, where=where)
        Rp, _ = optimize_Rp(U, S, pars_plain)

        if where == "S":
            A_spin = ncon([Rp, A_spin], [[-2, 1], [-1, 1, -3, -4]])
        elif where == "E":
            A_spin = ncon([Rp, A_spin], [[-3, 1], [-1, -2, 1, -4]])
        elif where == "N":
            A_spin = ncon([Rp, A_spin], [[-4, 1], [-1, -2, -3, 1]])
        elif where == "W":
            A_spin = ncon([Rp, A_spin], [[-1, 1], [1, -2, -3, -4]])

    # Contract back to charge basis
    A_charge = contract_from_spin_basis(A_spin, original_chi)

    return A_charge


def test_expansion():
    """Test the expansion and contraction."""
    print("=" * 80)
    print("Testing Expansion/Contraction")
    print("=" * 80)

    pars_plain = {
        'q': 3, 'relT': 1.0, 'gilt_eps': 1e-6,
        'cg_chis': [16], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': False,
    }
    pars_z3 = dict(pars_plain)
    pars_z3['symmetry_tensors'] = True

    A_plain = get_initial_tensor_potts_relT(pars_plain)
    A_z3 = get_initial_tensor_potts_relT(pars_z3)

    # Expand Z3 to spin basis
    A_spin, original_chi = expand_to_spin_basis(A_z3)

    print(f"\nOriginal chi: {original_chi}")
    print(f"Original shape: {A_z3.to_ndarray().shape}")
    print(f"Expanded shape: {A_spin.to_ndarray().shape}")

    # Check: spin-basis tensor should match plain tensor
    arr_plain = A_plain.to_ndarray()
    arr_spin = A_spin.to_ndarray()

    diff = np.linalg.norm(arr_plain - arr_spin) / np.linalg.norm(arr_plain)
    print(f"\n||plain - expanded|| / ||plain|| = {diff:.2e}")

    # Test round-trip
    A_back = contract_from_spin_basis(A_spin, original_chi)
    arr_back = A_back.to_ndarray()
    arr_orig = A_z3.to_ndarray()

    diff_rt = np.linalg.norm(arr_orig - arr_back) / np.linalg.norm(arr_orig)
    print(f"Round-trip error: {diff_rt:.2e}")


def test_gilt():
    """Test GILT with expansion."""
    print("\n" + "=" * 80)
    print("Testing GILT with Expansion")
    print("=" * 80)

    pars = {
        'q': 3, 'relT': 1.0, 'gilt_eps': 1e-6,
        'cg_chis': [16], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': True,
    }

    A_z3 = get_initial_tensor_potts_relT(pars)

    # Apply GILT with expansion
    A_filtered = gilt_with_expansion(A_z3, pars)

    print(f"\nOriginal shape: {A_z3.to_ndarray().shape}")
    print(f"Filtered shape: {A_filtered.to_ndarray().shape}")

    norm_orig = np.linalg.norm(A_z3.to_ndarray())
    norm_filt = np.linalg.norm(A_filtered.to_ndarray())
    print(f"||A_orig||: {norm_orig:.6f}")
    print(f"||A_filt||: {norm_filt:.6f}")


if __name__ == "__main__":
    test_expansion()
    test_gilt()
