#!/usr/bin/env python3
"""
Debug the GILT Rp filter for Z3 vs plain tensors.

Rp = sum_n t'[n] * U[:,:,n]†

The filter Rp should be close to identity when GILT converges.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D import get_envspec, optimize_Rp, build_Rp
from GiltTNR2D_Potts import get_initial_tensor_potts_relT

def main():
    print("=" * 70)
    print("Analyzing GILT Rp Filter: Z3 vs Plain")
    print("=" * 70)

    pars_plain = {
        'q': 3,
        'relT': 1.0,
        'gilt_eps': 1e-6,
        'cg_chis': [16],
        'symmetry_tensors': False,
    }
    pars_z3 = dict(pars_plain)
    pars_z3['symmetry_tensors'] = True

    A_plain = get_initial_tensor_potts_relT(pars_plain)
    A_z3 = get_initial_tensor_potts_relT(pars_z3)

    # Get environment spectra
    U_plain, S_plain = get_envspec(A_plain, A_plain, pars_plain, where="S")
    U_z3, S_z3 = get_envspec(A_z3, A_z3, pars_z3, where="S")

    print("\n--- Building Rp filters ---")

    # Plain
    Rp_plain, err_plain = optimize_Rp(U_plain, S_plain, pars_plain)
    arr_Rp_plain = Rp_plain.to_ndarray()
    print(f"\nPlain Rp:")
    print(f"  Shape: {arr_Rp_plain.shape}")
    print(f"  Insertion error: {err_plain:.6e}")
    print(f"  Rp matrix:\n{arr_Rp_plain}")

    # How close to identity?
    I_plain = np.eye(arr_Rp_plain.shape[0])
    dist_to_I_plain = np.linalg.norm(arr_Rp_plain - I_plain, 'fro')
    print(f"  ||Rp - I||_F: {dist_to_I_plain:.6f}")

    # Eigenvalues of Rp
    eig_Rp_plain = np.linalg.eigvalsh(arr_Rp_plain)
    print(f"  Eigenvalues: {eig_Rp_plain}")

    # Z3
    Rp_z3, err_z3 = optimize_Rp(U_z3, S_z3, pars_z3)
    arr_Rp_z3 = Rp_z3.to_ndarray()
    print(f"\nZ3 Rp:")
    print(f"  Shape: {arr_Rp_z3.shape}")
    print(f"  Insertion error: {err_z3:.6e}")
    print(f"  Rp matrix:\n{arr_Rp_z3}")

    # How close to identity?
    I_z3 = np.eye(arr_Rp_z3.shape[0])
    dist_to_I_z3 = np.linalg.norm(arr_Rp_z3 - I_z3, 'fro')
    print(f"  ||Rp - I||_F: {dist_to_I_z3:.6f}")

    # Eigenvalues of Rp
    eig_Rp_z3 = np.linalg.eigvalsh(arr_Rp_z3)
    print(f"  Eigenvalues: {eig_Rp_z3}")

    # Compare
    print("\n--- Comparison ---")
    print(f"Plain ||Rp - I||: {dist_to_I_plain:.6f}")
    print(f"Z3    ||Rp - I||: {dist_to_I_z3:.6f}")

    # The SVD of Rp tells us how it will modify the bond
    print("\n--- SVD of Rp ---")
    U_Rp_p, S_Rp_p, Vh_Rp_p = np.linalg.svd(arr_Rp_plain)
    print(f"Plain Rp singular values: {S_Rp_p}")

    U_Rp_z, S_Rp_z, Vh_Rp_z = np.linalg.svd(arr_Rp_z3)
    print(f"Z3 Rp singular values:    {S_Rp_z}")

    # Check the filtered traces tp
    print("\n--- Filtered traces tp ---")

    # Compute t and tp manually
    t_plain = ncon(U_plain, [1, 1, -1]).to_ndarray()
    S_plain_arr = S_plain.to_ndarray()

    gilt_eps = pars_plain['gilt_eps']
    ratio_plain = S_plain_arr / gilt_eps
    weight_plain = ratio_plain**2 / (1 + ratio_plain**2)
    tp_plain = t_plain * weight_plain

    print(f"Plain t:      {t_plain[:5]}")
    print(f"Plain weight: {weight_plain[:5]}")
    print(f"Plain tp:     {tp_plain[:5]}")

    t_z3 = ncon(U_z3, [1, 1, -1]).to_ndarray()
    S_z3_arr = S_z3.to_ndarray()

    ratio_z3 = S_z3_arr / gilt_eps
    weight_z3 = ratio_z3**2 / (1 + ratio_z3**2)
    tp_z3 = t_z3 * weight_z3

    print(f"\nZ3 t:      {t_z3[:5]}")
    print(f"Z3 weight: {weight_z3[:5]}")
    print(f"Z3 tp:     {tp_z3[:5]}")

    # The key difference: the trace structure
    print("\n--- Key difference analysis ---")
    print("Plain: Only t[0] is non-zero (identity-like mode)")
    print("Z3:    Both t[0] and t[1] are non-zero!")
    print("\nThis means Z3's GILT filter will act on TWO modes instead of one.")
    print("The extra mode might be filtering something that shouldn't be filtered.")

if __name__ == "__main__":
    main()
