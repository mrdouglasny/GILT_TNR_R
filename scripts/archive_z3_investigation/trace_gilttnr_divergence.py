#!/usr/bin/env python3
"""
Trace through GILT-TNR step by step to find where TensorZ3 vs plain diverge.

Key finding from previous test: TensorZ3.svd() gives correct singular values!
So the issue is elsewhere in the GILT-TNR pipeline.

Candidates:
1. GILT filtering (Rp application)
2. Truncation dimension allocation
3. Tensor contraction operations
"""

import sys
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr/GiltTNR')
sys.path.insert(0, '/workspaces/rg/ekrgilttrnr')

import numpy as np
from GiltTNR2D_Potts import get_initial_tensor_potts, get_initial_tensor_potts_z3
from tensors.symmetrytensors import TensorZ3
from tensors.tensor import Tensor

def build_dft_matrix(N):
    """Build N×N DFT matrix."""
    omega = np.exp(2j * np.pi / N)
    F = np.array([[omega**(s*q) for q in range(N)] for s in range(N)]) / np.sqrt(N)
    return F

def tensor_to_spin_basis(A_z3, N=3):
    """Convert TensorZ3 to spin basis array."""
    F = build_dft_matrix(N)
    Fdag = F.conj().T
    arr = A_z3.to_ndarray()

    chi = arr.shape[0]
    if chi == N:
        return np.einsum('ai,bj,ijkl,kc,ld->abcd', Fdag, Fdag, arr, F, F)
    else:
        block_size = chi // N
        F_large = np.kron(np.eye(block_size), F)
        Fdag_large = F_large.conj().T
        return np.einsum('ai,bj,ijkl,kc,ld->abcd', Fdag_large, Fdag_large, arr, F_large, F_large)

def compare_tensors_in_spin_basis(A_z3, A_plain, label=""):
    """Compare two tensors by converting to spin basis."""
    if hasattr(A_z3, 'sects'):
        arr_z3 = tensor_to_spin_basis(A_z3)
    else:
        arr_z3 = A_z3 if isinstance(A_z3, np.ndarray) else A_z3.to_ndarray()

    if hasattr(A_plain, 'sects'):
        arr_plain = tensor_to_spin_basis(A_plain)
    else:
        arr_plain = A_plain if isinstance(A_plain, np.ndarray) else A_plain.to_ndarray()

    if arr_z3.shape != arr_plain.shape:
        print(f"{label}: Shape mismatch Z3={arr_z3.shape} vs Plain={arr_plain.shape}")
        return None

    diff = np.linalg.norm(arr_z3 - arr_plain) / np.linalg.norm(arr_plain)
    print(f"{label}: Relative difference = {diff:.2e}")
    return diff

def trace_gilttnr_step():
    """Trace through one GILT-TNR step."""
    from GiltTNR2D import gilttnr_step

    print("="*70)
    print("Tracing GILT-TNR step")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))

    # Get initial tensors
    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    print("\nInitial tensors:")
    compare_tensors_in_spin_basis(A_z3, A_plain, "Initial")

    pars_z3 = {
        "gilt_eps": 1e-7,
        "cg_chis": list(range(2, 25)),
        "cg_eps": 1e-10,
        "verbosity": 2,  # Increase verbosity
        "symmetry_tensors": True
    }
    pars_plain = dict(pars_z3)
    pars_plain["symmetry_tensors"] = False

    print("\n" + "-"*50)
    print("Running Z3 GILT-TNR step with verbosity=2:")
    print("-"*50)

    try:
        A_z3_new, log_z3 = gilttnr_step(A_z3, 0.0, pars_z3)
        print(f"\nZ3 result shape: {A_z3_new.shape}")
    except Exception as e:
        print(f"Z3 step failed: {e}")
        import traceback
        traceback.print_exc()
        A_z3_new = None

    print("\n" + "-"*50)
    print("Running plain GILT-TNR step with verbosity=2:")
    print("-"*50)

    A_plain_new, log_plain = gilttnr_step(A_plain, 0.0, pars_plain)
    print(f"\nPlain result shape: {A_plain_new.shape}")

    if A_z3_new is not None:
        print("\n" + "-"*50)
        print("Comparing results:")
        print("-"*50)
        compare_tensors_in_spin_basis(A_z3_new, A_plain_new, "After step 1")

def trace_gilt_filtering():
    """Trace the GILT filtering step specifically."""
    from GiltTNR2D import gilt_plaq, apply_gilt, get_envspec, build_Rp, optimize_Rp

    print("\n" + "="*70)
    print("Tracing GILT filtering")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))

    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    pars = {
        "gilt_eps": 1e-7,
        "cg_chis": list(range(2, 25)),
        "cg_eps": 1e-10,
        "verbosity": 0,
    }

    print("\n1. Getting environment spectrum (gilt_plaq):")

    # For Z3
    print("\n   Z3 version:")
    U_z3, S_z3, envspec_z3, sumssq_z3 = gilt_plaq(A_z3, A_z3, pars)
    print(f"   envspec_z3 shape: U={U_z3.shape if hasattr(U_z3, 'shape') else type(U_z3)}")
    print(f"   S_z3: {S_z3[:10] if len(S_z3) > 10 else S_z3}")

    # For plain
    print("\n   Plain version:")
    U_plain, S_plain, envspec_plain, sumssq_plain = gilt_plaq(A_plain, A_plain, pars)
    print(f"   envspec_plain shape: U={U_plain.shape if hasattr(U_plain, 'shape') else type(U_plain)}")
    print(f"   S_plain: {S_plain[:10] if len(S_plain) > 10 else S_plain}")

    # Compare singular values
    print("\n   Comparing singular values:")
    S_z3_sorted = np.sort(np.array(S_z3))[::-1]
    S_plain_sorted = np.sort(np.array(S_plain))[::-1]
    min_len = min(len(S_z3_sorted), len(S_plain_sorted))
    diff = np.linalg.norm(S_z3_sorted[:min_len] - S_plain_sorted[:min_len])
    print(f"   ||S_z3 - S_plain|| = {diff:.2e}")

    print("\n2. Computing traces (envspec):")

    # Extract traces from envspec
    # envspec is a list of (S, U, t, sumssq) tuples for each leg

    for i, (leg, (S, U, t, ss)) in enumerate(zip(['l', 'r', 'u', 'd'], envspec_z3)):
        t_arr = np.array(t)
        print(f"   Z3 leg {leg}: {len(t_arr)} traces, top 5 |t| = {np.sort(np.abs(t_arr))[::-1][:5]}")

    for i, (leg, (S, U, t, ss)) in enumerate(zip(['l', 'r', 'u', 'd'], envspec_plain)):
        t_arr = np.array(t)
        print(f"   Plain leg {leg}: {len(t_arr)} traces, top 5 |t| = {np.sort(np.abs(t_arr))[::-1][:5]}")

def trace_trg_step():
    """Trace the TRG step (without GILT)."""
    from GiltTNR2D import trg

    print("\n" + "="*70)
    print("Tracing TRG step (no GILT)")
    print("="*70)

    beta_c = np.log(1 + np.sqrt(3))

    A_z3 = get_initial_tensor_potts_z3({"beta": beta_c})
    A_plain = get_initial_tensor_potts({"beta": beta_c})

    pars_z3 = {
        "cg_chis": list(range(2, 25)),
        "cg_eps": 1e-10,
        "verbosity": 2,
        "symmetry_tensors": True
    }
    pars_plain = dict(pars_z3)
    pars_plain["symmetry_tensors"] = False

    print("\nInitial:")
    compare_tensors_in_spin_basis(A_z3, A_plain, "Initial")

    print("\n" + "-"*50)
    print("TRG step on Z3:")
    print("-"*50)

    A_z3_trg, log_z3 = trg(A_z3, A_z3, 0.0, pars_z3)
    print(f"Z3 TRG result shape: {A_z3_trg.shape}")

    print("\n" + "-"*50)
    print("TRG step on plain:")
    print("-"*50)

    A_plain_trg, log_plain = trg(A_plain, A_plain, 0.0, pars_plain)
    print(f"Plain TRG result shape: {A_plain_trg.shape}")

    print("\n" + "-"*50)
    print("Comparing results:")
    print("-"*50)
    compare_tensors_in_spin_basis(A_z3_trg, A_plain_trg, "After TRG")

if __name__ == "__main__":
    trace_gilttnr_step()
    trace_trg_step()
