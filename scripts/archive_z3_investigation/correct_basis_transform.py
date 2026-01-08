#!/usr/bin/env python3
"""
Build the correct basis transformation between charge and spin bases.

Key insight: Singular values match → tensors contain same information
The difference is basis representation, not content!

For the initial 3×3×3×3 tensor:
- Spin basis: A[s1,s2,s3,s4] with s_i ∈ {0,1,2}
- Charge basis: A[q1,q2,q3,q4] with q_i ∈ {0,1,2}
- Relation: s_i = F @ q_i where F is DFT matrix

For the 9×9×9×9 tensor after TRG:
- Spin basis: A[S1,S2,S3,S4] with S_i ∈ {0,...,8}
  S encodes (s1,s2) coarse-grained pair as S = 3*s1 + s2
- Charge basis: A[(Q1,m1),(Q2,m2),(Q3,m3),(Q4,m4)]
  Q = total charge = (q1+q2) mod 3
  m = multiplicity index within sector

The transformation is block-structured based on how charges combine.
"""

import sys
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr/GiltTNR')
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr')

import numpy as np
from GiltTNR2D import trg
from GiltTNR2D_Potts import get_initial_tensor_potts, get_initial_tensor_potts_z3

def build_dft_matrix(N):
    """Build N×N DFT matrix: F[s,q] = ω^{sq}/√N where ω = e^{2πi/N}"""
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega**(s*q) for q in range(N)] for s in range(N)]) / np.sqrt(N)
    return F

def verify_initial_tensors():
    """Verify the transformation works for initial tensors."""
    print("="*70)
    print("Verifying initial tensor transformation")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))
    N = 3

    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    arr_charge = A_z3.to_ndarray()
    arr_spin = A_plain.to_ndarray()

    print(f"Charge basis shape: {arr_charge.shape}")
    print(f"Spin basis shape: {arr_spin.shape}")

    # Build DFT transformation
    F = build_dft_matrix(N)
    Fdag = F.conj().T

    # Transform charge to spin: A_spin = F† @ A_charge @ F (for each leg pair)
    # Actually: A_spin[s1,s2,s3,s4] = Σ_{q1,q2,q3,q4} F†[s1,q1] F†[s2,q2] A[q1,q2,q3,q4] F[q3,s3] F[q4,s4]
    arr_transformed = np.einsum('ia,jb,abcd,ck,dl->ijkl', Fdag, Fdag, arr_charge, F, F)

    diff = np.linalg.norm(arr_transformed - arr_spin) / np.linalg.norm(arr_spin)
    print(f"Relative difference after transformation: {diff:.2e}")

    # Check a few elements
    print("\nComparing specific elements:")
    print("  Original charge A[0,0,0,0]:", arr_charge[0,0,0,0])
    print("  Transformed A'[0,0,0,0]:", arr_transformed[0,0,0,0])
    print("  Plain spin A[0,0,0,0]:", arr_spin[0,0,0,0])

    return diff < 1e-10

def build_coarse_transform(N=3, dims_per_sector=3):
    """
    Build transformation for coarse-grained tensor (after TRG).

    After one TRG step:
    - Each coarse leg represents 2 fine legs
    - Spin index S ∈ {0,...,8} = {(s1,s2): s1,s2 ∈ {0,1,2}} with S = 3*s1 + s2
    - Charge index (Q, m) where Q = (q1+q2) mod 3 and m ∈ {0,1,2} is multiplicity

    The transformation maps:
    - Charge basis index i ∈ {0,...,8} ordered as [Q=0 sector, Q=1 sector, Q=2 sector]
    - Spin basis index S ∈ {0,...,8}

    For each (S, Q, m), we need the transformation coefficient.
    """

    total_dim = N * dims_per_sector  # = 9

    # First, figure out the ordering of charge basis indices
    # TensorZ3 with shape [[3,3,3],...] has indices ordered by sector
    # i = sector_offset[Q] + m where sector_offset[Q] = Q * dims_per_sector

    # The spin basis has S = 3*s1 + s2 for (s1, s2) pairs
    # Each pair (s1, s2) has total charge (s1 + s2) mod 3

    # For each charge Q, there are 3 spin pairs with that charge:
    # Q=0: (0,0), (1,2), (2,1) → S = 0, 5, 7
    # Q=1: (0,1), (1,0), (2,2) → S = 1, 3, 8
    # Q=2: (0,2), (1,1), (2,0) → S = 2, 4, 6

    print("\nBuilding coarse-grained transformation:")

    # Map from (S) to (Q, m) where m is the order within the charge sector
    spin_to_charge = {}
    charge_to_spins = {0: [], 1: [], 2: []}

    for s1 in range(N):
        for s2 in range(N):
            S = N * s1 + s2
            Q = (s1 + s2) % N
            charge_to_spins[Q].append(S)

    print("Charge to spins mapping:")
    for Q, spins in charge_to_spins.items():
        print(f"  Q={Q}: spins={spins}")

    # Build the transformation matrix
    # T[S, i] where S is spin index and i is charge-basis index
    # i = Q * dims_per_sector + m

    # The transformation relates fine basis to coarse:
    # For spin, we need to transform EACH of the two fine legs independently
    # |S⟩ = |s1, s2⟩
    # In charge basis: |q1, q2⟩
    # So |S⟩ = Σ_{q1,q2} (F†[s1,q1] F†[s2,q2]) |q1, q2⟩

    # But the coarse tensor's charge basis has combined charges Q = (q1+q2) mod 3
    # The multiplicity m tells us WHICH (q1, q2) pair within that charge

    # For Q=0: (0,0), (1,2), (2,1) - these have m=0,1,2
    # For Q=1: (0,1), (1,0), (2,2)
    # For Q=2: (0,2), (1,1), (2,0)

    charge_to_pairs = {0: [(0,0), (1,2), (2,1)],
                       1: [(0,1), (1,0), (2,2)],
                       2: [(0,2), (1,1), (2,0)]}

    print("\nCharge to (q1,q2) pairs mapping:")
    for Q, pairs in charge_to_pairs.items():
        print(f"  Q={Q}: pairs={pairs}")

    # Build full transformation matrix
    F = build_dft_matrix(N)
    Fdag = F.conj().T

    T = np.zeros((total_dim, total_dim), dtype=complex)

    for S in range(total_dim):
        s1, s2 = S // N, S % N
        for Q in range(N):
            for m, (q1, q2) in enumerate(charge_to_pairs[Q]):
                i = Q * dims_per_sector + m
                # Coefficient: F†[s1,q1] * F†[s2,q2]
                T[S, i] = Fdag[s1, q1] * Fdag[s2, q2]

    print(f"\nTransformation matrix T shape: {T.shape}")
    print(f"T is unitary: {np.allclose(T @ T.conj().T, np.eye(total_dim))}")

    return T

def verify_trg_transformation():
    """Verify the transformation works for TRG output."""
    print("\n" + "="*70)
    print("Verifying TRG output transformation")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))
    N = 3

    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    pars_z3 = {"cg_chis": list(range(2, 25)), "cg_eps": 1e-10, "symmetry_tensors": True, "verbosity": 0}
    pars_plain = {"cg_chis": list(range(2, 25)), "cg_eps": 1e-10, "symmetry_tensors": False, "verbosity": 0}

    A_z3_trg, _ = trg(A_z3, A_z3, 0.0, pars_z3)
    A_plain_trg, _ = trg(A_plain, A_plain, 0.0, pars_plain)

    arr_charge = A_z3_trg.to_ndarray()
    arr_spin = A_plain_trg.to_ndarray()

    print(f"Charge basis (Z3) shape: {arr_charge.shape}")
    print(f"Spin basis (plain) shape: {arr_spin.shape}")

    # Build transformation
    T = build_coarse_transform(N=3, dims_per_sector=3)
    Tdag = T.conj().T

    # Transform: A_spin[S1,S2,S3,S4] = Σ T[S1,i1] T[S2,i2] A_charge[i1,i2,i3,i4] T†[i3,S3] T†[i4,S4]
    arr_transformed = np.einsum('Ii,Jj,ijkl,kK,lL->IJKL', T, T, arr_charge, Tdag, Tdag)

    diff = np.linalg.norm(arr_transformed - arr_spin) / np.linalg.norm(arr_spin)
    print(f"\nRelative difference after transformation: {diff:.2e}")

    # If still large, maybe the multiplicity ordering is wrong
    # Let's check what ordering TensorZ3 actually uses

    print("\nChecking TensorZ3 sector structure:")
    print(f"Sectors: {list(A_z3_trg.sects.keys())[:5]}...")  # Show first 5

    # The sectors tell us the charge assignment
    # Shape [[3,3,3],...] means each leg has 3 dims in each of 3 sectors

    return diff

def debug_tensorz3_ordering():
    """Debug the internal ordering of TensorZ3."""
    print("\n" + "="*70)
    print("Debugging TensorZ3 internal ordering")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))
    N = 3

    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})

    pars_z3 = {"cg_chis": list(range(2, 25)), "cg_eps": 1e-10, "symmetry_tensors": True, "verbosity": 0}
    A_z3_trg, _ = trg(A_z3, A_z3, 0.0, pars_z3)

    print(f"TRG output shape: {A_z3_trg.shape}")

    # Look at specific sectors
    print("\nNon-zero sectors (showing first few):")
    for i, (qvec, block) in enumerate(sorted(A_z3_trg.sects.items())):
        if i >= 10:
            print("  ...")
            break
        q0, q1, q2, q3 = qvec
        val = block[0, 0, 0, 0] if block.shape == (1,1,1,1) else f"block{block.shape}"
        print(f"  {qvec}: {val}")

    # The to_ndarray() flattens this to a dense array
    # Let's understand the index mapping

    arr = A_z3_trg.to_ndarray()

    # For a [[3,3,3]] shape leg:
    # Index i ∈ {0,...,8}
    # Sector q = i // 3 (if dims are [3,3,3])
    # Within-sector index m = i % 3

    print("\nIndex to sector mapping for [[3,3,3]] leg:")
    for i in range(9):
        q = i // 3
        m = i % 3
        print(f"  i={i}: sector q={q}, multiplicity m={m}")

if __name__ == "__main__":
    if verify_initial_tensors():
        print("\n✓ Initial tensor transformation works!")
    else:
        print("\n✗ Initial tensor transformation failed!")

    verify_trg_transformation()
    debug_tensorz3_ordering()
