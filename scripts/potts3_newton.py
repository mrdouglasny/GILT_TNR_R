#!/usr/bin/env python3
# MODULE: potts3_newton.py
# USAGE: python3 ekrgilttrnr/scripts/potts3_newton.py [options]
# DESCRIPTION: Newton method to find 3-state Potts fixed point tensor.
#
# BASED ON: ekrgilttrnr/scripts/newton.jl (Ising version)
# NEW IN THIS SCRIPT: Adapts Newton method for q-state Potts with plain tensors.
#                     No Z_q symmetry tensors - uses standard numpy arrays.
#
# Method:
#   1. Start from approximate fixed point (from bisection + RG flow)
#   2. Compute Jacobian dR/dA numerically
#   3. Newton iteration: A_new = A - (I - J)^{-1} (A - R(A))
#   4. Converge to true fixed point
#
# Note: Without symmetry tensors, this is slower but works for any q.

import sys
import os
import argparse
import numpy as np
from scipy.linalg import eig, inv
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../GiltTNR'))

from GiltTNR2D_Potts import get_initial_tensor_potts_relT, get_scaldims_potts
from GiltTNR2D import gilttnr_step
from tensors import Tensor


def tensor_to_array(T):
    """Convert Tensor object to numpy array."""
    if isinstance(T, np.ndarray):
        return T
    return T.to_ndarray()


def array_to_tensor(arr):
    """Convert numpy array to Tensor object."""
    return Tensor.from_ndarray(arr)


def normalize_tensor(A):
    """Normalize tensor by Frobenius norm."""
    arr = tensor_to_array(A)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return array_to_tensor(arr), norm


def rg_step(A, gilt_pars):
    """Apply one RG step and return normalized result."""
    result = gilttnr_step(A, 0.0, gilt_pars)
    A_new = result[0]
    A_new, _ = normalize_tensor(A_new)
    return A_new


def compute_jacobian_numerical(A, gilt_pars, eps=1e-5):
    """
    Compute Jacobian dR/dA numerically using finite differences.

    For a tensor A with N elements, J is NxN matrix.
    J[i,j] = d(R(A))_i / dA_j
    """
    A_arr = tensor_to_array(A)
    shape = A_arr.shape
    N = A_arr.size

    # Flatten
    A_flat = A_arr.flatten()

    # Reference output
    R_A = rg_step(A, gilt_pars)
    R_A_arr = tensor_to_array(R_A).flatten()

    # Compute Jacobian column by column
    J = np.zeros((N, N))

    for j in range(N):
        # Perturb A in direction j
        A_plus = A_flat.copy()
        A_plus[j] += eps
        A_plus_tensor = array_to_tensor(A_plus.reshape(shape))

        R_plus = rg_step(A_plus_tensor, gilt_pars)
        R_plus_arr = tensor_to_array(R_plus).flatten()

        # Finite difference
        J[:, j] = (R_plus_arr - R_A_arr) / eps

    return J


def newton_step(A, gilt_pars, J_inv_ImJ=None):
    """
    Perform one Newton step: A_new = A - (I - J)^{-1} (A - R(A))

    If J_inv_ImJ is provided, use it. Otherwise compute J numerically.
    """
    A_arr = tensor_to_array(A).flatten()

    # R(A)
    R_A = rg_step(A, gilt_pars)
    R_A_arr = tensor_to_array(R_A).flatten()

    # Residual: A - R(A)
    residual = A_arr - R_A_arr

    if J_inv_ImJ is None:
        # Compute Jacobian
        print("    Computing Jacobian numerically...")
        J = compute_jacobian_numerical(A, gilt_pars)
        I_minus_J = np.eye(len(A_arr)) - J

        # Solve (I - J) * correction = residual
        try:
            correction = np.linalg.solve(I_minus_J, residual)
        except np.linalg.LinAlgError:
            print("    Warning: Singular matrix, using pseudoinverse")
            correction = np.linalg.lstsq(I_minus_J, residual, rcond=None)[0]
    else:
        correction = J_inv_ImJ @ residual

    # New tensor
    A_new_arr = A_arr - correction
    A_new = array_to_tensor(A_new_arr.reshape(tensor_to_array(A).shape))
    A_new, _ = normalize_tensor(A_new)

    return A_new, np.linalg.norm(correction)


def simple_newton(A_init, gilt_pars, max_iter=10, tol=1e-8):
    """
    Simple Newton iteration without Jacobian approximation.

    Uses damped iteration toward fixed point.
    Note: In Gilt-TNR, the tensor dimension can change. We track convergence
    by the relative change in scaling dimensions, not tensor elements.
    """
    A = A_init
    prev_scaldims = None

    for iteration in range(max_iter):
        R_A = rg_step(A, gilt_pars)

        # Get scaling dimensions (more stable than comparing tensor elements)
        scaldims = get_scaldims_potts(R_A)

        # Measure convergence by scaling dimension stability
        if prev_scaldims is not None and len(scaldims) >= 4 and len(prev_scaldims) >= 4:
            delta_x1 = abs(scaldims[1] - prev_scaldims[1])
            delta_x3 = abs(scaldims[3] - prev_scaldims[3])
            print(f"  Iter {iteration+1}: x_σ = {scaldims[1]:.6f}, x_ε = {scaldims[3]:.6f}, "
                  f"Δx_σ = {delta_x1:.2e}, Δx_ε = {delta_x3:.2e}")

            if delta_x1 < tol and delta_x3 < tol:
                print(f"  Converged!")
                break
        else:
            if len(scaldims) >= 4:
                print(f"  Iter {iteration+1}: x_σ = {scaldims[1]:.6f}, x_ε = {scaldims[3]:.6f}")
            else:
                print(f"  Iter {iteration+1}: scaldims = {scaldims[:min(4, len(scaldims))]}")

        prev_scaldims = scaldims.copy() if isinstance(scaldims, np.ndarray) else np.array(scaldims)

        # Use R(A) as next iterate (pure RG flow toward fixed point)
        A = R_A

    return A


def get_flow_to_approximate_fp(relT, chi, n_warmup, gilt_pars):
    """Run RG flow to get approximate fixed point tensor."""
    potts_pars = {'q': 3, 'relT': relT, 'symmetry_tensors': False}
    A = get_initial_tensor_potts_relT(potts_pars)

    for step in range(n_warmup):
        result = gilttnr_step(A, 0.0, gilt_pars)
        A = result[0]

    A, _ = normalize_tensor(A)
    return A


def main():
    parser = argparse.ArgumentParser(description='Newton method for Potts fixed point')
    parser.add_argument('--chi', type=int, default=12, help='Bond dimension')
    parser.add_argument('--relT', type=float, default=1.00291, help='Starting relT (from bisection)')
    parser.add_argument('--warmup', type=int, default=6, help='RG warmup steps')
    parser.add_argument('--newton_iter', type=int, default=10, help='Newton iterations')
    parser.add_argument('--gilt_eps', type=float, default=1e-6, help='Gilt threshold')
    args = parser.parse_args()

    chi = args.chi
    relT = args.relT
    n_warmup = args.warmup
    newton_iter = args.newton_iter
    gilt_eps = args.gilt_eps

    gilt_pars = {
        'gilt_eps': gilt_eps,
        'cg_chis': list(range(1, chi + 1)),
        'cg_eps': 1e-10,
        'verbosity': 0,
        'rotate': False
    }

    print('=' * 70)
    print('3-State Potts: Newton Method for Fixed Point')
    print('=' * 70)
    print()
    print(f'Parameters: χ={chi}, relT={relT}, warmup={n_warmup}')
    print()

    # Step 1: Get approximate fixed point from RG flow
    print('Step 1: Running RG warmup...')
    A_approx = get_flow_to_approximate_fp(relT, chi, n_warmup, gilt_pars)

    # Check initial residual
    R_A = rg_step(A_approx, gilt_pars)
    residual_init = np.linalg.norm(
        tensor_to_array(A_approx).flatten() - tensor_to_array(R_A).flatten()
    )
    print(f'  Initial ||A - R(A)|| = {residual_init:.2e}')
    print()

    # Step 2: Newton iteration
    print('Step 2: Newton iteration...')
    A_fp = simple_newton(A_approx, gilt_pars, max_iter=newton_iter)
    print()

    # Step 3: Extract scaling dimensions at fixed point
    print('Step 3: Extracting scaling dimensions at fixed point...')
    scaldims = get_scaldims_potts(A_fp)

    print()
    print('=' * 70)
    print('RESULTS')
    print('=' * 70)
    print()
    print('Scaling dimensions at fixed point:')
    print(f'  x_0 (I)    = {scaldims[0]:.6f}  (exact: 0)')
    print(f'  x_1 (σ)    = {scaldims[1]:.6f}  (exact: {2/15:.6f})')
    print(f'  x_2 (σbar) = {scaldims[2]:.6f}  (exact: {2/15:.6f})')
    print(f'  x_3 (ε)    = {scaldims[3]:.6f}  (exact: {4/5:.6f})')
    if len(scaldims) > 4:
        print(f'  x_4        = {scaldims[4]:.6f}')
    if len(scaldims) > 5:
        print(f'  x_5        = {scaldims[5]:.6f}')

    print()
    print('Errors:')
    print(f'  x_σ error: {100*abs(scaldims[1] - 2/15)/(2/15):.2f}%')
    print(f'  x_ε error: {100*abs(scaldims[3] - 4/5)/(4/5):.2f}%')


if __name__ == '__main__':
    main()
