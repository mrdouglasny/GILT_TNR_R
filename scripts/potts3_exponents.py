#!/usr/bin/env python3
# MODULE: potts3_exponents.py
# USAGE: python3 ekrgilttrnr/scripts/potts3_exponents.py [options]
# INPUTS: None (constructs tensor from parameters)
# OUTPUTS:
#   - stdout: scaling dimensions at each RG step
#   - ekrgilttrnr/data/potts3_exponents/<config>.dat
# DESCRIPTION: Extract scaling dimensions for 3-state Potts model using Gilt-TNR.
#
# BASED ON: ekrgilttrnr/GiltTNR/GiltTNR2D_Ising_benchmarks.py
# NEW IN THIS SCRIPT: Applies Gilt-TNR to 3-state Potts model for critical exponent extraction.
#
# 3-STATE POTTS CFT (c=4/5) EXPECTED VALUES:
#   x_identity = 0
#   x_spin (σ) = 2/15 ≈ 0.1333
#   x_energy (ε) = 4/5 = 0.8
#   x_spin2 = 4/3 ≈ 1.333
#
# RUNTIME: ~5-10 min for χ=16, ~30 min for χ=24

import sys
import os
import argparse
import numpy as np

# Add GiltTNR to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../GiltTNR'))

from GiltTNR2D_Potts import get_initial_tensor_potts, get_initial_tensor_potts_relT, get_scaldims_potts
from GiltTNR2D import gilttnr_step


def main():
    parser = argparse.ArgumentParser(description='3-state Potts scaling dimensions via Gilt-TNR')
    parser.add_argument('--q', type=int, default=3, help='Number of Potts states')
    parser.add_argument('--relT', type=float, default=1.0, help='Relative temperature (1.0 = critical)')
    parser.add_argument('--chi', type=int, default=16, help='Bond dimension')
    parser.add_argument('--steps', type=int, default=30, help='Number of RG steps')
    parser.add_argument('--gilt_eps', type=float, default=1e-6, help='Gilt truncation threshold')
    parser.add_argument('--output', type=str, default='', help='Output file')
    args = parser.parse_args()

    q = args.q
    relT = args.relT
    chi = args.chi
    max_steps = args.steps
    gilt_eps = args.gilt_eps

    # Default output file
    output_file = args.output
    if output_file == '':
        os.makedirs('ekrgilttrnr/data/potts3_exponents', exist_ok=True)
        output_file = f'ekrgilttrnr/data/potts3_exponents/q{q}_chi{chi}_relT{relT}.dat'

    # Critical point
    beta_c = np.log(1 + np.sqrt(q))

    print('=' * 70)
    print('3-State Potts Model: Scaling Dimension Extraction via Gilt-TNR')
    print('=' * 70)
    print()
    print('Parameters:')
    print(f'  q (states):     {q}')
    print(f'  relT:           {relT} (1.0 = critical)')
    print(f'  β_c (exact):    {beta_c:.6f}')
    print(f'  β (actual):     {beta_c/relT:.6f}')
    print(f'  χ (bond dim):   {chi}')
    print(f'  Gilt ε:         {gilt_eps}')
    print(f'  RG steps:       {max_steps}')
    print()
    print('Expected CFT scaling dimensions (c=4/5):')
    print('  x_identity = 0')
    print('  x_spin (σ) = 2/15 ≈ 0.1333')
    print('  x_energy (ε) = 4/5 = 0.8')
    print()
    print('-' * 70)

    # Build initial tensor
    potts_pars = {
        'q': q,
        'relT': relT,
        'symmetry_tensors': False
    }

    gilt_pars = {
        'gilt_eps': gilt_eps,
        'cg_chis': list(range(1, chi + 1)),
        'cg_eps': 1e-10,
        'verbosity': 0,
        'rotate': False
    }

    print('Building initial tensor...')
    A = get_initial_tensor_potts_relT(potts_pars)
    log_fact = 0.0

    # Storage for results
    results = []

    print()
    print(f'{"Step":<6}  {"x_1":<10}  {"x_2":<10}  {"x_3":<10}  {"x_4":<10}  {"x_5":<10}')
    print('-' * 70)

    for step in range(1, max_steps + 1):
        # RG Step (GiltTNR2D.gilttnr_step returns (A, log_fact))
        result = gilttnr_step(A, log_fact, gilt_pars)
        A = result[0]
        log_fact = result[1]

        # Extract scaling dimensions
        scaldims = get_scaldims_potts(A)

        # Pad to 6 elements (first is identity = 0)
        while len(scaldims) < 6:
            scaldims = np.append(scaldims, np.nan)

        # x_0 = 0 (identity), so x_1 is first non-trivial
        x_vals = scaldims[1:6]

        results.append({'step': step, 'x_vals': x_vals})

        print(f'{step:<6}  {x_vals[0]:<10.6f}  {x_vals[1]:<10.6f}  {x_vals[2]:<10.6f}  {x_vals[3]:<10.6f}  {x_vals[4]:<10.6f}')

    print('-' * 70)
    print()

    # Final analysis
    if len(results) > 5:
        # Average over last few steps
        last_n = min(5, len(results))
        x1_vals = [r['x_vals'][0] for r in results[-last_n:]]
        x2_vals = [r['x_vals'][1] for r in results[-last_n:]]
        x1_avg = np.mean(x1_vals)
        x2_avg = np.mean(x2_vals)

        x_spin_exact = 2/15
        x_energy_exact = 4/5

        print(f'Final scaling dimensions (averaged over last {last_n} steps):')
        print(f'  x_1 = {x1_avg:.6f}  (expected σ: {x_spin_exact:.6f}, error: {100*abs(x1_avg - x_spin_exact)/x_spin_exact:.2f}%)')
        print(f'  x_2 = {x2_avg:.6f}  (expected ε: {x_energy_exact:.6f}, error: {100*abs(x2_avg - x_energy_exact)/x_energy_exact:.2f}%)')

    # Save results to file
    print()
    print(f'Saving results to {output_file}')
    with open(output_file, 'w') as f:
        f.write('# step x_1 x_2 x_3 x_4 x_5\n')
        for r in results:
            f.write(f"{r['step']} {' '.join(f'{x:.10f}' for x in r['x_vals'])}\n")

    print()
    print('=' * 70)
    print('COMPLETED')
    print('=' * 70)


if __name__ == '__main__':
    main()
