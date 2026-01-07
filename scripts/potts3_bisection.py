#!/usr/bin/env python3
# MODULE: potts3_bisection.py
# USAGE: python3 ekrgilttrnr/scripts/potts3_bisection.py [options]
# DESCRIPTION: Binary search to find the critical relT for 3-state Potts model.
#
# BASED ON: potts3_exponents.py
# NEW IN THIS SCRIPT: Uses bisection to find relT where flow stays near fixed point.
#
# Method:
#   - At T < T_c (relT < relT_c): flow → ordered (x → 0)
#   - At T > T_c (relT > relT_c): flow → disordered (x → ∞)
#   - At T = T_c: flow → fixed point (x → x_σ, x_ε constant)
#
# We track whether x_1 increases or decreases after several RG steps.

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../GiltTNR'))

from GiltTNR2D_Potts import get_initial_tensor_potts_relT, get_scaldims_potts
from GiltTNR2D import gilttnr_step


def run_flow(relT, chi, n_steps, gilt_eps=1e-6, verbose=False):
    """Run RG flow and return scaling dimensions at each step."""
    potts_pars = {'q': 3, 'relT': relT, 'symmetry_tensors': False}
    gilt_pars = {
        'gilt_eps': gilt_eps,
        'cg_chis': list(range(1, chi + 1)),
        'cg_eps': 1e-10,
        'verbosity': 0,
        'rotate': False
    }

    A = get_initial_tensor_potts_relT(potts_pars)
    log_fact = 0.0

    x1_history = []
    x2_history = []

    for step in range(1, n_steps + 1):
        result = gilttnr_step(A, log_fact, gilt_pars)
        A = result[0]
        log_fact = result[1]

        scaldims = get_scaldims_potts(A)
        x1 = scaldims[1] if len(scaldims) > 1 else np.nan
        x2 = scaldims[2] if len(scaldims) > 2 else np.nan

        x1_history.append(x1)
        x2_history.append(x2)

        if verbose:
            print(f"  Step {step}: x_1 = {x1:.6f}, x_2 = {x2:.6f}")

    return x1_history, x2_history


def classify_flow(x1_history, threshold_step=6):
    """
    Classify flow direction based on x_1 behavior.

    Returns:
        'ordered': x_1 decreasing (T < T_c)
        'disordered': x_1 increasing (T > T_c)
        'critical': x_1 roughly constant
    """
    if len(x1_history) < threshold_step + 2:
        return 'unknown'

    x1_early = x1_history[threshold_step - 1]  # Value at step threshold_step
    x1_late = x1_history[threshold_step + 1]   # Value 2 steps later

    if np.isnan(x1_early) or np.isnan(x1_late):
        return 'unknown'

    delta = x1_late - x1_early
    relative_delta = delta / max(x1_early, 0.01)

    if relative_delta < -0.1:
        return 'ordered'
    elif relative_delta > 0.1:
        return 'disordered'
    else:
        return 'critical'


def bisection_search(chi, n_steps, tol=1e-4, max_iter=20, gilt_eps=1e-6):
    """Binary search for critical relT."""

    # Initial bounds: relT < 1 is ordered (lower T), relT > 1 is disordered (higher T)
    # Wait - for Potts, higher T means disordered
    # relT = T/T_c, so relT > 1 means T > T_c (disordered)

    relT_low = 1.0     # Start at exact critical (might be ordered side)
    relT_high = 1.003  # Slightly above (disordered side based on earlier tests)

    # First, verify bounds
    print(f"Testing bounds...")
    print(f"  relT = {relT_low}:")
    x1_low, _ = run_flow(relT_low, chi, n_steps, gilt_eps)
    flow_low = classify_flow(x1_low)
    print(f"    Flow: {flow_low}")

    print(f"  relT = {relT_high}:")
    x1_high, _ = run_flow(relT_high, chi, n_steps, gilt_eps)
    flow_high = classify_flow(x1_high)
    print(f"    Flow: {flow_high}")

    if flow_low == flow_high:
        print(f"Warning: both bounds give same flow direction ({flow_low})")
        print("Adjusting bounds...")

        # Try wider range
        if flow_low == 'ordered':
            relT_high = 1.01
        else:
            relT_low = 0.995

        x1_high, _ = run_flow(relT_high, chi, n_steps, gilt_eps)
        flow_high = classify_flow(x1_high)
        x1_low, _ = run_flow(relT_low, chi, n_steps, gilt_eps)
        flow_low = classify_flow(x1_low)

        print(f"  New relT_low = {relT_low}: {flow_low}")
        print(f"  New relT_high = {relT_high}: {flow_high}")

    print()
    print("Starting bisection...")
    print("-" * 60)

    for iteration in range(max_iter):
        relT_mid = (relT_low + relT_high) / 2

        print(f"Iter {iteration+1}: relT = {relT_mid:.8f} ", end="")
        x1_mid, x2_mid = run_flow(relT_mid, chi, n_steps, gilt_eps)
        flow_mid = classify_flow(x1_mid)

        # Report x values at a fixed step
        step_report = min(5, len(x1_mid))
        x1_val = x1_mid[step_report-1] if step_report > 0 else np.nan
        x2_val = x2_mid[step_report-1] if step_report > 0 else np.nan

        print(f"→ {flow_mid:10s}  (x_1={x1_val:.4f}, x_2={x2_val:.4f} at step {step_report})")

        if flow_mid == 'ordered':
            relT_low = relT_mid
        elif flow_mid == 'disordered':
            relT_high = relT_mid
        else:
            print(f"  Found approximately critical point!")
            break

        if relT_high - relT_low < tol:
            print(f"  Converged to tolerance {tol}")
            break

    relT_c = (relT_low + relT_high) / 2
    return relT_c, x1_mid, x2_mid


def main():
    parser = argparse.ArgumentParser(description='Bisection search for Potts critical point')
    parser.add_argument('--chi', type=int, default=16, help='Bond dimension')
    parser.add_argument('--steps', type=int, default=10, help='RG steps per evaluation')
    parser.add_argument('--tol', type=float, default=1e-5, help='Convergence tolerance')
    parser.add_argument('--gilt_eps', type=float, default=1e-6, help='Gilt threshold')
    args = parser.parse_args()

    print('=' * 70)
    print('3-State Potts: Bisection Search for Critical Point')
    print('=' * 70)
    print()
    print(f'Parameters: χ={args.chi}, steps={args.steps}, tol={args.tol}')
    print()
    print('CFT predictions (c=4/5):')
    print(f'  x_σ = 2/15 = {2/15:.6f}')
    print(f'  x_ε = 4/5 = {4/5:.6f}')
    print()

    relT_c, x1_final, x2_final = bisection_search(
        args.chi, args.steps, args.tol, gilt_eps=args.gilt_eps
    )

    print()
    print('=' * 70)
    print('RESULTS')
    print('=' * 70)

    beta_c_exact = np.log(1 + np.sqrt(3))
    beta_c_est = beta_c_exact / relT_c

    print(f'Estimated relT_c = {relT_c:.8f}')
    print(f'Estimated β_c = {beta_c_est:.6f}')
    print(f'Exact β_c = {beta_c_exact:.6f}')
    print(f'Error = {100*abs(beta_c_est - beta_c_exact)/beta_c_exact:.4f}%')
    print()

    # Final scaling dimensions at best relT
    print(f'Scaling dimensions at relT = {relT_c:.8f}:')
    for i, (x1, x2) in enumerate(zip(x1_final[:8], x2_final[:8])):
        print(f'  Step {i+1}: x_1 = {x1:.6f}, x_2 = {x2:.6f}')

    print()
    print(f'Best x_σ estimate (step 4-5 avg): {np.mean(x1_final[3:5]):.6f} (exact: {2/15:.6f})')
    print(f'Best x_ε estimate (step 4-5 avg): {np.mean(x2_final[3:5]):.6f} (exact: {4/5:.6f})')


if __name__ == '__main__':
    main()
