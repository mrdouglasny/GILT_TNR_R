#!/usr/bin/env python3
"""
Verify the root cause hypothesis for Z3 GILT instability.

Tests:
1. Do plain tensor singular values have degeneracies that split in Z3?
2. What happens if we DON'T filter the "spurious" Z3 modes?
3. Are the extra Z3 modes actually identity-like or something else?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D import gilttnr_step, get_envspec
from GiltTNR2D_Potts import get_initial_tensor_potts_relT

def test_singular_value_degeneracy():
    """Test 1: Check if plain SVD has degeneracies."""
    print("=" * 70)
    print("Test 1: Singular Value Degeneracy")
    print("=" * 70)

    chi = 16
    pars_plain = {
        'q': 3, 'relT': 1.0, 'gilt_eps': 1e-6,
        'cg_chis': [chi], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': False,
    }
    pars_z3 = dict(pars_plain)
    pars_z3['symmetry_tensors'] = True

    A_plain = get_initial_tensor_potts_relT(pars_plain)
    A_z3 = get_initial_tensor_potts_relT(pars_z3)

    # Get environment spectra
    U_plain, S_plain = get_envspec(A_plain, A_plain, pars_plain, where="S")
    U_z3, S_z3 = get_envspec(A_z3, A_z3, pars_z3, where="S")

    S_plain_arr = S_plain.to_ndarray()
    S_z3_arr = S_z3.to_ndarray()

    print(f"\nPlain singular values (top 15):")
    for i, s in enumerate(S_plain_arr[:15]):
        print(f"  S[{i}] = {s:.6f}")

    print(f"\nZ3 singular values (top 15):")
    for i, s in enumerate(S_z3_arr[:15]):
        print(f"  S[{i}] = {s:.6f}")

    # Check for near-degeneracies in plain
    print("\nDegeneracy check (plain): |S[i] - S[i+1]| / S[i]")
    for i in range(min(10, len(S_plain_arr)-1)):
        if S_plain_arr[i] > 1e-10:
            rel_gap = abs(S_plain_arr[i] - S_plain_arr[i+1]) / S_plain_arr[i]
            deg_marker = " <-- DEGENERATE" if rel_gap < 0.01 else ""
            print(f"  Gap[{i},{i+1}] = {rel_gap:.6f}{deg_marker}")


def test_mode_structure():
    """Test 2: Analyze what the 'extra' Z3 modes actually represent."""
    print("\n" + "=" * 70)
    print("Test 2: What Are The Extra Z3 Modes?")
    print("=" * 70)

    chi = 16
    pars_plain = {
        'q': 3, 'relT': 1.0, 'gilt_eps': 1e-6,
        'cg_chis': [chi], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': False,
    }
    pars_z3 = dict(pars_plain)
    pars_z3['symmetry_tensors'] = True

    A_plain_0 = get_initial_tensor_potts_relT(pars_plain)
    A_z3_0 = get_initial_tensor_potts_relT(pars_z3)

    # One step
    A_plain_1, _ = gilttnr_step(A_plain_0, 0.0, pars_plain)
    A_z3_1, _ = gilttnr_step(A_z3_0, 0.0, pars_z3)

    U_plain, S_plain = get_envspec(A_plain_1, A_plain_1, pars_plain, where="S")
    U_z3, S_z3 = get_envspec(A_z3_1, A_z3_1, pars_z3, where="S")

    # Compute traces
    t_plain = ncon(U_plain, [1, 1, -1]).to_ndarray()
    t_z3 = ncon(U_z3, [1, 1, -1]).to_ndarray()

    arr_U_plain = U_plain.to_ndarray()
    arr_U_z3 = U_z3.to_ndarray()

    chi_plain = arr_U_plain.shape[0]
    chi_z3 = arr_U_z3.shape[0]

    print(f"\nComparing modes with significant trace in Z3 but not plain:")
    print(f"(chi_plain={chi_plain}, chi_z3={chi_z3})")

    # Find modes where Z3 has significant trace but plain doesn't
    for i in range(min(10, len(t_z3))):
        t_z3_val = abs(t_z3[i])
        t_plain_val = abs(t_plain[i]) if i < len(t_plain) else 0

        if t_z3_val > 0.3 and t_plain_val < 0.1:
            print(f"\n  Mode {i}: |t_z3|={t_z3_val:.4f}, |t_plain|={t_plain_val:.4f}")

            # Analyze this U_z3 mode
            U_mode = arr_U_z3[:, :, i]
            print(f"    U_z3 mode shape: {U_mode.shape}")
            print(f"    Frobenius norm: {np.linalg.norm(U_mode):.4f}")
            print(f"    Trace: {np.trace(U_mode):.4f}")

            # Check if it's close to identity
            I = np.eye(U_mode.shape[0])
            dist_to_I = np.linalg.norm(U_mode - I * np.trace(U_mode) / U_mode.shape[0])
            print(f"    Distance to scaled identity: {dist_to_I:.4f}")

            # Check block structure
            diag = np.diag(U_mode)
            off_diag_norm = np.linalg.norm(U_mode - np.diag(diag))
            print(f"    Diagonal elements: {diag[:5]}...")
            print(f"    Off-diagonal norm: {off_diag_norm:.6f}")


def test_transform_consistency():
    """Test 3: Check if Z3 and plain give same physics in spin basis."""
    print("\n" + "=" * 70)
    print("Test 3: Transform Z3 Back to Spin Basis")
    print("=" * 70)

    chi = 16
    pars_plain = {
        'q': 3, 'relT': 1.0, 'gilt_eps': 1e-6,
        'cg_chis': [chi], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': False,
    }
    pars_z3 = dict(pars_plain)
    pars_z3['symmetry_tensors'] = True

    A_plain = get_initial_tensor_potts_relT(pars_plain)
    A_z3 = get_initial_tensor_potts_relT(pars_z3)

    arr_plain = A_plain.to_ndarray()
    arr_z3 = A_z3.to_ndarray()

    # DFT matrix
    omega = np.exp(2j * np.pi / 3)
    F = np.array([
        [1, 1, 1],
        [1, omega, omega**2],
        [1, omega**2, omega]
    ]) / np.sqrt(3)
    Fdag = F.T.conj()

    # Transform Z3 back to spin basis: T_spin = F† ⊗ F† ⊗ F ⊗ F T_charge
    # For 4-leg tensor: T_spin[a,b,c,d] = sum_{i,j,k,l} F†[a,i] F†[b,j] F[c,k] F[d,l] T_charge[i,j,k,l]
    arr_z3_to_spin = np.einsum('ai,bj,ck,dl,ijkl->abcd', Fdag, Fdag, F, F, arr_z3)

    # Compare
    diff = np.linalg.norm(arr_plain - arr_z3_to_spin)
    norm_plain = np.linalg.norm(arr_plain)

    print(f"\n||T_plain - F†⊗F†⊗F⊗F T_z3||: {diff:.2e}")
    print(f"||T_plain||: {norm_plain:.4f}")
    print(f"Relative difference: {diff/norm_plain:.2e}")

    if diff/norm_plain < 1e-10:
        print("\n✓ Z3 tensor correctly transforms back to plain tensor")
        print("  This confirms the tensor construction is correct.")
    else:
        print("\n✗ Mismatch! There may be an issue with tensor construction.")


def test_rg_step_comparison():
    """Test 4: Compare RG step in detail."""
    print("\n" + "=" * 70)
    print("Test 4: Compare RG Steps in Detail")
    print("=" * 70)

    chi = 16
    pars_plain = {
        'q': 3, 'relT': 1.0, 'gilt_eps': 0,  # NO GILT filtering
        'cg_chis': [chi], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': False,
    }
    pars_z3 = dict(pars_plain)
    pars_z3['symmetry_tensors'] = True

    A_plain_0 = get_initial_tensor_potts_relT(pars_plain)
    A_z3_0 = get_initial_tensor_potts_relT(pars_z3)

    print("\nRunning RG WITHOUT GILT (gilt_eps=0):")

    # Multiple steps without GILT
    A_plain = A_plain_0
    A_z3 = A_z3_0
    log_plain = 0.0
    log_z3 = 0.0

    for step in range(1, 5):
        A_plain, log_plain = gilttnr_step(A_plain, log_plain, pars_plain)
        A_z3, log_z3 = gilttnr_step(A_z3, log_z3, pars_z3)

        # Compare tensor norms instead (simpler)
        arr_plain = A_plain.to_ndarray()
        arr_z3 = A_z3.to_ndarray()

        norm_plain = np.linalg.norm(arr_plain)
        norm_z3 = np.linalg.norm(arr_z3)

        # Shapes
        shape_plain = arr_plain.shape
        shape_z3 = arr_z3.shape

        print(f"  Step {step}: shape_plain={shape_plain}, shape_z3={shape_z3}")
        print(f"           norm_plain={norm_plain:.4e}, norm_z3={norm_z3:.4e}")


def main():
    test_singular_value_degeneracy()
    test_mode_structure()
    test_transform_consistency()
    test_rg_step_comparison()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Key questions to answer:
1. Are there degeneracies in plain that split in Z3? → Check Test 1
2. What do the 'extra' Z3 modes represent? → Check Test 2
3. Is the tensor construction correct? → Check Test 3
4. Does Z3 diverge even WITHOUT GILT? → Check Test 4

If Test 4 shows Z3 diverges without GILT, then the root cause is NOT
the GILT trace computation, but something in the coarse-graining itself.
""")


if __name__ == "__main__":
    main()
