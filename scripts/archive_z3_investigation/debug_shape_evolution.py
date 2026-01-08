#!/usr/bin/env python3
"""
Track shape evolution to understand Z3 instability.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from tensors import Tensor, TensorZ2, TensorZ3
from GiltTNR2D import gilttnr_step
from GiltTNR2D_Potts import get_initial_tensor_potts_relT, get_scaldims_potts

def total_dim(shape):
    """Total dimension of tensor (product of leg dimensions)."""
    if isinstance(shape[0], list):
        return tuple(sum(s) for s in shape)
    else:
        return shape

def main():
    print("=" * 70)
    print("Shape Evolution: Z3 vs Plain")
    print("=" * 70)

    chi = 16
    n_steps = 5

    pars_z3 = {
        'q': 3,
        'relT': 1.0,
        'gilt_eps': 1e-6,
        'cg_chis': [chi],
        'cg_eps': 1e-10,
        'verbosity': 0,
        'symmetry_tensors': True,
    }

    pars_plain = dict(pars_z3)
    pars_plain['symmetry_tensors'] = False

    A_z3 = get_initial_tensor_potts_relT(pars_z3)
    A_plain = get_initial_tensor_potts_relT(pars_plain)

    log_z3, log_plain = 0.0, 0.0

    print(f"\n{'Step':>4} {'Z3 shape':>30} {'Plain shape':>20} {'Z3 sectors':>15}")
    print("-" * 75)

    for step in range(n_steps + 1):
        z3_shape = A_z3.shape if hasattr(A_z3, 'shape') else None
        z3_total = total_dim(z3_shape) if z3_shape else None
        plain_shape = A_plain.to_ndarray().shape

        # Show per-sector breakdown for Z3
        if hasattr(A_z3, 'shape'):
            # Get number of singular values per sector
            sector_dims = {}
            for k, v in A_z3.sects.items():
                if k[0] not in sector_dims:
                    sector_dims[k[0]] = 0
                sector_dims[k[0]] += v.shape[0] if len(v.shape) > 0 else 1

        print(f"{step:>4} {str(z3_total):>30} {str(plain_shape):>20} {str(z3_shape[0]):>15}")

        if step < n_steps:
            A_z3, log_z3 = gilttnr_step(A_z3, log_z3, pars_z3)
            A_plain, log_plain = gilttnr_step(A_plain, log_plain, pars_plain)

    # Show detailed sector analysis at step 2 (where divergence starts)
    print("\n" + "=" * 70)
    print("Detailed Analysis at Step 2 (where divergence starts)")
    print("=" * 70)

    # Reset
    A_z3 = get_initial_tensor_potts_relT(pars_z3)
    A_plain = get_initial_tensor_potts_relT(pars_plain)
    log_z3, log_plain = 0.0, 0.0

    for step in range(2):
        A_z3, log_z3 = gilttnr_step(A_z3, log_z3, pars_z3)
        A_plain, log_plain = gilttnr_step(A_plain, log_plain, pars_plain)

    print("\nZ3 tensor after step 2:")
    print(f"  shape: {A_z3.shape}")
    print(f"  qhape: {A_z3.qhape}")
    print(f"  dirs: {A_z3.dirs}")

    # Count elements in each charge sector
    sector_sizes = {}
    for k, v in A_z3.sects.items():
        charge = k[0]
        if charge not in sector_sizes:
            sector_sizes[charge] = 0
        sector_sizes[charge] += v.size
    print(f"  Elements per charge (leg 0): {sector_sizes}")

    print("\nPlain tensor after step 2:")
    arr = A_plain.to_ndarray()
    print(f"  shape: {arr.shape}")

    # Check singular values for each
    print("\n" + "=" * 70)
    print("Singular Value Spectrum Comparison")
    print("=" * 70)

    # Get SVD spectrum
    from ncon import ncon

    # Z3 spectrum
    S_z3 = A_z3.svd([0,1], [2,3])[1].to_ndarray()
    S_z3 = -np.sort(-S_z3)
    S_z3 /= S_z3[0]
    print(f"\nZ3 singular values (normalized): {S_z3[:10]}")

    # Plain spectrum
    S_plain = A_plain.svd([0,1], [2,3])[1].to_ndarray()
    S_plain = -np.sort(-S_plain)
    S_plain /= S_plain[0]
    print(f"Plain singular values (normalized): {S_plain[:10]}")

    # Check ratio of first to second
    print(f"\nZ3 S[0]/S[1] = {S_z3[0]/S_z3[1]:.4f}")
    print(f"Plain S[0]/S[1] = {S_plain[0]/S_plain[1]:.4f}")

if __name__ == "__main__":
    main()
