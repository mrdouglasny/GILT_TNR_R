#!/usr/bin/env python3
"""
Test GILT filtering alone (no coarse-graining) on Z3 vs Plain.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from GiltTNR2D import gilttnr_step, apply_gilt
from GiltTNR2D_Potts import get_initial_tensor_potts_relT, get_scaldims_potts

def main():
    print("=" * 70)
    print("GILT-Only Test: Z3 vs Plain")
    print("=" * 70)

    chi = 16

    base_pars = {
        'q': 3,
        'relT': 1.0,
        'gilt_eps': 1e-6,
        'cg_chis': [chi],
        'cg_eps': 1e-10,
        'verbosity': 1,  # Enable verbose output
    }

    print("\n1. Plain Tensor GILT:")
    print("-" * 70)
    pars_plain = dict(base_pars)
    pars_plain['symmetry_tensors'] = False
    A_plain = get_initial_tensor_potts_relT(pars_plain)
    print(f"Initial shape: {A_plain.to_ndarray().shape}")

    # Apply GILT only
    A_plain_gilt = apply_gilt(A_plain, pars_plain)
    print(f"After GILT shape: {A_plain_gilt.to_ndarray().shape}")

    sd_plain = get_scaldims_potts(A_plain_gilt)
    print(f"Scaling dims: {sd_plain[:5]}")

    print("\n2. TensorZ3 GILT:")
    print("-" * 70)
    pars_z3 = dict(base_pars)
    pars_z3['symmetry_tensors'] = True
    A_z3 = get_initial_tensor_potts_relT(pars_z3)
    print(f"Initial shape: {A_z3.shape}")

    # Apply GILT only
    A_z3_gilt = apply_gilt(A_z3, pars_z3)
    print(f"After GILT shape: {A_z3_gilt.shape}")

    sd_z3 = get_scaldims_potts(A_z3_gilt)
    print(f"Scaling dims: {sd_z3[:5]}")


if __name__ == "__main__":
    main()
