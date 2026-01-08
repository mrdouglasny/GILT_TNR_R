#!/usr/bin/env python3
"""
Debug sector dimension allocation during Z3 flow.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from GiltTNR2D import gilttnr_step
from GiltTNR2D_Potts import get_initial_tensor_potts_relT

def analyze_tensor(A, label):
    """Analyze sector dimensions of a tensor."""
    print(f"\n{label}:")
    if not hasattr(A, 'qhape'):
        print("  (plain tensor, no sectors)")
        arr = A.to_ndarray()
        print(f"  shape: {arr.shape}")
        return

    print(f"  shape: {A.shape}")
    print(f"  qhape: {A.qhape}")

    # Analyze per-leg sector distribution
    for leg in range(len(A.shape)):
        dims = A.shape[leg]
        qs = A.qhape[leg]
        print(f"  leg {leg}: dims={dims}, qhape={qs}, total={sum(dims)}")


def main():
    print("=" * 70)
    print("Sector Dimension Analysis During Z3 Flow")
    print("=" * 70)

    chi = 16
    n_steps = 3

    base_pars = {
        'q': 3,
        'relT': 1.0,
        'gilt_eps': 1e-6,
        'cg_chis': [chi],
        'cg_eps': 1e-10,
        'verbosity': 0,
        'symmetry_tensors': True,
    }

    for balanced, name in [(False, "Greedy"), (True, "Balanced")]:
        print(f"\n{'='*70}")
        print(f"{name} Allocation (balanced_sectors={balanced})")
        print("=" * 70)

        pars = dict(base_pars)
        pars['balanced_sectors'] = balanced

        A = get_initial_tensor_potts_relT(pars)
        log_fact = 0.0

        analyze_tensor(A, f"Step 0 (initial)")

        for step in range(1, n_steps + 1):
            try:
                A, log_fact = gilttnr_step(A, log_fact, pars)
                analyze_tensor(A, f"Step {step}")
            except Exception as e:
                print(f"\n  Step {step} failed: {e}")
                break


if __name__ == "__main__":
    main()
