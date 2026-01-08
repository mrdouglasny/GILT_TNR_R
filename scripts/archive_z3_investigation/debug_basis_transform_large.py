#!/usr/bin/env python3
"""
Debug basis transformation for larger bond dimensions.

After TRG step:
- Z3 shape: [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
  = 3+3+3 = 9 dims per leg, arranged as [sector0, sector1, sector2]
- Plain shape: (9, 9, 9, 9)
  = 9 dims per leg in spin basis

The basis transformation needs to handle the block structure correctly.
"""

import sys
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr/GiltTNR')
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr')

import numpy as np
from GiltTNR2D import trg
from GiltTNR2D_Potts import get_initial_tensor_potts, get_initial_tensor_potts_z3
from tensors.symmetrytensors import TensorZ3
from tensors.tensor import Tensor

def build_dft_matrix(N):
    """Build N×N DFT matrix."""
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega**(s*q) for q in range(N)] for s in range(N)]) / np.sqrt(N)
    return F

def analyze_z3_structure(A_z3):
    """Analyze the structure of a TensorZ3 tensor."""
    print(f"TensorZ3 shape: {A_z3.shape}")
    print(f"Number of sectors: {len(A_z3.sects)}")

    # Get sector dimensions per leg
    leg_dims = A_z3.shape
    print(f"Dimensions per leg (as list of sector dims): {leg_dims}")

    # Total dims per leg
    total_per_leg = [sum(dims) for dims in leg_dims]
    print(f"Total dims per leg: {total_per_leg}")

    # The to_ndarray() should give a dense array
    arr = A_z3.to_ndarray()
    print(f"Dense array shape: {arr.shape}")

    return arr

def correct_basis_transform(A_z3, N=3):
    """
    Correct basis transformation for TensorZ3 with multi-dimensional sectors.

    For a TensorZ3 with shape [[d0, d1, d2], ...]:
    - Total dimension per leg = d0 + d1 + d2
    - The dense array has indices ordered by charge: first d0 with q=0, then d1 with q=1, etc.

    The transformation matrix F relates spin and charge bases:
    - In charge basis: index i in sector q has charge q
    - In spin basis: index s runs from 0 to total_dim-1

    For each sector with d_q dimensions, we get d_q copies of the charge-q state.
    The transformation is block-diagonal:
    F_large = diag(F_q0 ⊗ I_{d0}, F_q1 ⊗ I_{d1}, F_q2 ⊗ I_{d2})

    Wait, that's not quite right either. Let me think more carefully...

    Actually, the Z3 tensor in charge basis has:
    - Index (q, m) where q ∈ {0,1,2} is charge and m ∈ {0,...,d_q-1} is multiplicity
    - The dense array flattens this to index i = sum(d_k for k<q) + m

    The spin basis has:
    - Index s ∈ {0,...,total_dim-1}

    The transformation should be:
    A_spin[s1, s2, s3, s4] = Σ A_charge[q1,m1,q2,m2,q3,m3,q4,m4] * F[s1, (q1,m1)] * ...

    But what is F[s, (q,m)]? The original 3×3 DFT relates spin and charge for a single
    degree of freedom. For multiple multiplicities, we need to understand what the
    spin basis index s means.

    Hmm, this is getting complicated. Let me try a different approach.
    """

    arr_charge = A_z3.to_ndarray()
    chi = arr_charge.shape[0]

    print(f"Attempting to transform {chi}×{chi}×{chi}×{chi} tensor")

    if chi == N:
        # Simple case - one dim per sector
        F = build_dft_matrix(N)
        Fdag = F.conj().T
        return np.einsum('ai,bj,ijkl,kc,ld->abcd', Fdag, Fdag, arr_charge, F, F)

    # Multi-dimensional case
    # The issue: how do multiplicities in charge basis map to spin basis?

    # For TRG output with shape [[3,3,3],...]:
    # - 3 sectors (q=0,1,2), each with 3 dimensions (m=0,1,2)
    # - Total 9 dimensions, same as plain tensor

    # The plain tensor in spin basis has 9 dimensions representing
    # different spin configurations on the coarse-grained lattice.

    # The Z3 tensor represents the SAME states, but organized by charge.
    # Each "multiplicity" m within sector q corresponds to a different
    # combination of spins that gives total charge q.

    # For the initial 3×3×3×3 tensor:
    # - Each leg has 3 spin values (0,1,2)
    # - Each spin value IS a charge value (spin = charge for single site)

    # After TRG doubling:
    # - Each coarse leg represents 2 fine legs
    # - Spin index s ∈ {0,...,8} = {(s1,s2): s1,s2 ∈ {0,1,2}}
    # - Charge q = (s1 + s2) mod 3

    # So the mapping is:
    # s = 3*s1 + s2 for s1,s2 ∈ {0,1,2}
    # q = (s1 + s2) mod 3
    # m = multiplicity index within charge sector

    # Let's build the explicit mapping
    sector_dims = A_z3.shape[0]  # [d0, d1, d2] for leg 0
    if not isinstance(sector_dims, list):
        sector_dims = list(sector_dims) if hasattr(sector_dims, '__iter__') else [sector_dims]

    print(f"Sector dims for leg 0: {sector_dims}")

    # For 3×3 coarse-graining with Z3 symmetry:
    # Each charge sector q has 3 states (from the 9/3 = 3 ways to get charge q)

    # Build the transformation matrix
    # T[(q,m), s] = 1/√3 * ω^{q*s_total} where s_total = s1 + s2 for s = 3*s1 + s2
    # Hmm, this still needs the correct normalization...

    # Actually, let's just check if the tensors are equal up to permutation
    return arr_charge

def check_numerical_agreement():
    """Check if Z3 and plain agree numerically after TRG."""
    print("="*70)
    print("Checking numerical agreement after TRG")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))

    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    pars_z3 = {"cg_chis": list(range(2, 25)), "cg_eps": 1e-10, "symmetry_tensors": True, "verbosity": 0}
    pars_plain = {"cg_chis": list(range(2, 25)), "cg_eps": 1e-10, "symmetry_tensors": False, "verbosity": 0}

    # TRG step
    A_z3_trg, _ = trg(A_z3, A_z3, 0.0, pars_z3)
    A_plain_trg, _ = trg(A_plain, A_plain, 0.0, pars_plain)

    print("\nZ3 after TRG:")
    arr_z3 = analyze_z3_structure(A_z3_trg)

    print("\nPlain after TRG:")
    arr_plain = A_plain_trg.to_ndarray()
    print(f"Plain array shape: {arr_plain.shape}")

    # Compare directly (should both be 9×9×9×9)
    print("\n" + "-"*50)
    print("Direct comparison of dense arrays:")
    print("-"*50)

    print(f"Z3 dense shape: {arr_z3.shape}")
    print(f"Plain dense shape: {arr_plain.shape}")

    if arr_z3.shape != arr_plain.shape:
        print("Shapes differ!")
        return

    diff = np.linalg.norm(arr_z3 - arr_plain) / np.linalg.norm(arr_plain)
    print(f"Relative difference: {diff:.2e}")

    if diff > 0.1:
        print("\nLarge difference! Let's analyze the structure...")

        # Check if it's a basis ordering issue
        # The TensorZ3 orders by charge: [q0 states, q1 states, q2 states]
        # The plain tensor orders by spin: [s=0, s=1, ..., s=8]

        print("\n1. Checking if singular values match:")
        mat_z3 = arr_z3.reshape(81, 81)
        mat_plain = arr_plain.reshape(81, 81)

        _, S_z3, _ = np.linalg.svd(mat_z3)
        _, S_plain, _ = np.linalg.svd(mat_plain)

        S_z3_sorted = np.sort(S_z3)[::-1]
        S_plain_sorted = np.sort(S_plain)[::-1]

        print(f"   Top 10 S_z3: {S_z3_sorted[:10]}")
        print(f"   Top 10 S_plain: {S_plain_sorted[:10]}")

        S_diff = np.linalg.norm(S_z3_sorted - S_plain_sorted) / np.linalg.norm(S_plain_sorted)
        print(f"   Relative difference in singular values: {S_diff:.2e}")

        print("\n2. Checking if it's a permutation/basis issue:")

        # For each leg, the Z3 tensor has indices organized by charge
        # Let's build the permutation that maps charge order to spin order

        # Spin index s in [0,8] corresponds to (s1, s2) where s = 3*s1 + s2
        # Each (s1, s2) has charge (s1 + s2) mod 3

        perm = []
        for q in range(3):  # For each charge sector
            for s in range(9):  # Find spins with this charge
                s1, s2 = s // 3, s % 3
                if (s1 + s2) % 3 == q:
                    perm.append(s)

        print(f"   Charge-to-spin permutation: {perm}")

        # Apply permutation to Z3 array
        arr_z3_reorder = arr_z3[np.ix_(perm, perm, perm, perm)]

        diff_reorder = np.linalg.norm(arr_z3_reorder - arr_plain) / np.linalg.norm(arr_plain)
        print(f"   After reordering: relative difference = {diff_reorder:.2e}")

def trace_individual_elements():
    """Trace specific tensor elements through TRG."""
    print("\n" + "="*70)
    print("Tracing individual tensor elements through TRG")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))

    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    arr_z3_init = A_z3.to_ndarray()
    arr_plain_init = A_plain.to_ndarray()

    print("\nInitial tensor comparison:")

    # Check specific elements
    for i in range(3):
        for j in range(3):
            val_z3 = arr_z3_init[i, j, 0, 0]
            val_plain = arr_plain_init[i, j, 0, 0]
            if abs(val_z3) > 1e-10 or abs(val_plain) > 1e-10:
                print(f"  A[{i},{j},0,0]: Z3={val_z3:.6f}, Plain={val_plain:.6f}")

if __name__ == "__main__":
    check_numerical_agreement()
    trace_individual_elements()
