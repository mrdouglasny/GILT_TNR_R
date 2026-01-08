#!/usr/bin/env python3
"""
Analyze GILT after truncation: This is where Z3 diverges.

Key insight from initial analysis:
- At step 0, both have same trace values but different U structure
- Plain U has off-diagonal elements
- Z3 U is purely diagonal

After truncation, does this difference cause problems?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from ncon import ncon
from GiltTNR2D import gilttnr_step, get_envspec, optimize_Rp
from GiltTNR2D_Potts import get_initial_tensor_potts_relT


def analyze_step(name, A, pars, step_num):
    """Analyze environment at a given step."""
    print(f"\n--- {name} at Step {step_num} ---")

    U, S = get_envspec(A, A, pars, where="S")

    arr_U = U.to_ndarray()
    arr_S = S.to_ndarray()
    chi = arr_U.shape[0]

    print(f"  χ = {chi}, #modes = {len(arr_S)}")

    # Compute traces
    t = ncon(U, [1, 1, -1]).to_ndarray()

    # Count non-zero trace modes
    n_nonzero = np.sum(np.abs(t) > 0.1)
    total_t = np.sum(np.abs(t))

    print(f"  Non-zero traces (|t| > 0.1): {n_nonzero}")
    print(f"  Total |t|: {total_t:.4f}")

    # Show trace distribution
    print(f"  Top traces: ", end="")
    sorted_idx = np.argsort(np.abs(t))[::-1]
    for i in range(min(5, len(t))):
        idx = sorted_idx[i]
        print(f"|t[{idx}]|={np.abs(t[idx]):.3f} ", end="")
    print()

    # Check off-diagonal structure of top U modes
    print(f"  U structure (off-diag ratio for top modes):")
    for i in range(min(3, arr_U.shape[2])):
        U_i = arr_U[:, :, i]
        diag = np.diag(np.diag(U_i))
        off_diag_ratio = np.linalg.norm(U_i - diag) / np.linalg.norm(U_i)
        print(f"    U[{i}]: off-diag/total = {off_diag_ratio:.4f}, |t|={np.abs(t[i]):.4f}")

    # Get Rp filter
    Rp, err = optimize_Rp(U, S, pars)
    arr_Rp = Rp.to_ndarray()

    # Check how much Rp differs from identity
    I = np.eye(arr_Rp.shape[0])
    dist_from_I = np.linalg.norm(arr_Rp - I)

    print(f"  ||Rp - I|| = {dist_from_I:.4f}")

    # SVD of Rp to see filtering strength
    s_Rp = np.linalg.svd(arr_Rp, compute_uv=False)
    print(f"  Rp singular values: min={s_Rp.min():.4f}, max={s_Rp.max():.4f}")

    return n_nonzero, total_t, dist_from_I


def main():
    print("=" * 70)
    print("GILT Analysis After Truncation")
    print("=" * 70)

    chi = 16
    gilt_eps = 1e-6

    pars_plain = {
        'q': 3, 'relT': 1.0, 'gilt_eps': gilt_eps,
        'cg_chis': [chi], 'cg_eps': 1e-10, 'verbosity': 0,
        'symmetry_tensors': False,
    }
    pars_z3 = dict(pars_plain)
    pars_z3['symmetry_tensors'] = True

    # Get initial tensors
    A_plain = get_initial_tensor_potts_relT(pars_plain)
    A_z3 = get_initial_tensor_potts_relT(pars_z3)

    # Step 0 (before any RG)
    print("\n" + "=" * 70)
    print("STEP 0 (Initial)")
    print("=" * 70)
    analyze_step("Plain", A_plain, pars_plain, 0)
    analyze_step("Z3", A_z3, pars_z3, 0)

    # Run steps
    for step in range(1, 4):
        print("\n" + "=" * 70)
        print(f"STEP {step}")
        print("=" * 70)

        A_plain, _ = gilttnr_step(A_plain, 0.0, pars_plain)
        A_z3, _ = gilttnr_step(A_z3, 0.0, pars_z3)

        n_p, t_p, d_p = analyze_step("Plain", A_plain, pars_plain, step)
        n_z, t_z, d_z = analyze_step("Z3", A_z3, pars_z3, step)

        print(f"\n  COMPARISON:")
        print(f"    Non-zero modes: Plain={n_p}, Z3={n_z}")
        print(f"    Total |t|: Plain={t_p:.4f}, Z3={t_z:.4f}")
        print(f"    ||Rp - I||: Plain={d_p:.4f}, Z3={d_z:.4f}")

        if d_z > 2 * d_p:
            print(f"    ⚠️  Z3 GILT filter is {d_z/d_p:.1f}x more aggressive!")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The key question: After truncation, does Z3 develop more "identity-like" modes
that get filtered by GILT?

If ||Rp - I|| grows faster for Z3 than plain, GILT is removing more from Z3.
This would cause the divergence we observe.
""")


if __name__ == "__main__":
    main()
