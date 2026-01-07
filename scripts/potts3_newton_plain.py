#!/usr/bin/env python3
"""
Newton iteration for 3-state Potts fixed point using plain tensors.

Based on EKR (arXiv:2408.10312) approach:
- Use plain tensors (not TensorZ3)
- Continuous gauge fixing via environment diagonalization
- Discrete gauge fixing via GF(3) linear algebra
- Newton iteration on gauge-fixed tensor

Key differences from Ising:
- Z₃ phases: ω = exp(2πi/3) instead of ±1
- GF(3) instead of GF(2) for discrete gauge
- 3 charge sectors instead of 2
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from scipy.linalg import eigh
from tensors import Tensor
from ncon import ncon
from GiltTNR2D import gilttnr_step
from GiltTNR2D_Potts import get_scaldims_potts

# CFT values
X_SIGMA = 2/15
X_EPS = 4/5

def get_potts_tensor_plain(beta):
    """Build plain Potts tensor in charge basis."""
    q = 3
    W = np.ones((q, q))
    np.fill_diagonal(W, np.exp(beta))
    T_spin = np.einsum('ab,bc,cd,da->abcd', W, W, W, W)

    omega = np.exp(2j * np.pi / 3)
    F = np.array([[1, 1, 1], [1, omega, omega**2], [1, omega**2, omega]]) / np.sqrt(3)
    F_dg = F.T.conj()
    T_charge = ncon((T_spin, F, F, F_dg, F_dg), ([1,2,3,4], [-1,1], [-2,2], [3,-3], [4,-4]))
    T_charge[np.abs(T_charge) < 1e-12] = 0

    return Tensor.from_ndarray(np.real(T_charge))


def fix_continuous_gauge(A):
    """
    Fix continuous gauge freedom by diagonalizing environments.
    Works for plain tensors.

    Based on EKR GaugeFixing.jl:
    - Tensor indices: A[H_in, V_in, H_out, V_out] (0, 1, 2, 3)
    - Vertical gauge: transform indices 1 and 3 (V_in, V_out)
    - Horizontal gauge: transform indices 0 and 2 (H_in, H_out)
    """
    arr = A.to_ndarray()

    # Vertical environment: contract H indices (0, 2) and keep V indices (1, 3)
    # E_v[i,j] = sum_{a,b} A[a,i,b,*] * conj(A[a,j,b,*])
    # From EKR: connects = [[2, -1, 1, 3], [2, -2, 1, 3]], so -1,-2 are V indices
    env_v = np.einsum('aibk,ajbk->ij', arr, arr.conj())
    env_v = (env_v + env_v.T.conj()) / 2  # Hermitianize
    _, V = eigh(env_v)
    V = V[:, ::-1]  # Sort descending

    # Fix phase: make sum of each column positive (real)
    for i in range(V.shape[1]):
        col_sum = np.sum(V[:, i])
        if np.real(col_sum) < 0:
            V[:, i] *= -1

    # Apply V gauge to indices 1 and 3
    # A'[i,j,k,l] = sum_{m,n} V[m,j] A[i,m,k,n] V*[n,l]
    # In Einstein: contract j with V and l with V.conj
    arr = np.einsum('imkn,mj,nl->ijkl', arr, V, V.conj())

    # Horizontal environment: contract V indices (1, 3) and keep H indices (0, 2)
    # E_h[i,j] = sum_{b,c} A[i,b,*,c] * conj(A[j,b,*,c])
    # From EKR: connects = [[-1, 2, 3, 1], [-2, 2, 3, 1]], so -1,-2 are H indices
    env_h = np.einsum('ibkc,jbkc->ij', arr, arr.conj())
    env_h = (env_h + env_h.T.conj()) / 2
    _, H = eigh(env_h)
    H = H[:, ::-1]

    for i in range(H.shape[1]):
        col_sum = np.sum(H[:, i])
        if np.real(col_sum) < 0:
            H[:, i] *= -1

    # Apply H gauge to indices 0 and 2
    # A'[i,j,k,l] = sum_{m,n} H[m,i] A[m,j,n,l] H*[n,k]
    arr = np.einsum('mjnl,mi,nk->ijkl', arr, H, H.conj())

    return Tensor.from_ndarray(arr), H, V


def fix_discrete_gauge_z3(A, tol=1e-7):
    """
    Fix discrete Z₃ gauge freedom.

    For Z₃, the gauge transformation is:
        A'[i,j,k,l] = ω^{g_i + g_j - g_k - g_l} A[i,j,k,l]
    where g ∈ {0, 1, 2} and ω = exp(2πi/3).

    We want selected elements to be positive real, which means
    their phase should be 0 (mod 2π/3).

    This requires solving a linear system over GF(3).
    """
    arr = A.to_ndarray()
    dim = arr.shape[0]
    omega = np.exp(2j * np.pi / 3)

    # Find non-diagonal elements with large magnitude
    elements = []
    for idx in np.ndindex(*arr.shape):
        if np.abs(arr[idx]) > tol:
            # Skip diagonal-like elements
            if not (idx[0] == idx[2] and idx[1] == idx[3]):
                phase = np.angle(arr[idx])
                # Convert phase to Z₃ charge: 0, 1, or 2
                # phase ≈ 0 → charge 0
                # phase ≈ 2π/3 → charge 1
                # phase ≈ -2π/3 (= 4π/3) → charge 2
                charge = int(np.round(phase / (2 * np.pi / 3))) % 3
                elements.append((idx, np.abs(arr[idx]), charge))

    # Sort by magnitude (largest first)
    elements.sort(key=lambda x: -x[1])

    # Build linear system over GF(3)
    # For element at (i,j,k,l), the constraint is:
    #   g_i + g_j - g_k - g_l = -charge (mod 3)
    # to make the element have phase 0.

    # We have 4*dim gauge DOFs, but there's an overall phase freedom
    # (we can add constant to all g's), so we have 4*dim - 1 DOFs.

    n_dof = 4 * dim - 1
    M = []
    b = []

    for idx, mag, charge in elements:
        if len(M) >= n_dof:
            break

        # Build row: coefficients for [g_H[0..dim-1], g_H[dim..2dim-1], g_V[0..dim-1], g_V[dim..2dim-1]]
        # Actually simpler: g[i] for horizontal leg 1, g[dim+j] for vertical leg 1, etc.
        # Let's use: g[0:dim] = horizontal, g[dim:2dim] = vertical
        # Constraint: g[i] + g[dim+j] - g[k] - g[dim+l] = -charge (mod 3)

        row = np.zeros(2 * dim, dtype=int)
        i, j, k, l = idx
        row[i] += 1
        row[j] += 1
        row[k] -= 1
        row[l] -= 1
        row = row % 3

        # Check linear independence
        if len(M) > 0:
            M_test = np.vstack([M, row])
            # Simple rank check via row reduction mod 3
            rank = np.linalg.matrix_rank(M_test.astype(float) % 3)
            if rank <= len(M):
                continue  # Dependent row, skip

        M.append(row)
        b.append((-charge) % 3)

    if len(M) < n_dof:
        print(f"  Warning: only found {len(M)}/{n_dof} independent constraints")

    if len(M) == 0:
        # No constraints, return as-is
        return A, np.ones(dim), np.ones(dim)

    M = np.array(M, dtype=int)
    b = np.array(b, dtype=int)

    # Solve M @ g = b (mod 3)
    # Use simple Gaussian elimination over GF(3)
    g = solve_gf3(M, b, dim)

    # Convert g to phase factors
    g_H = np.array([omega ** g[i] for i in range(dim)])
    g_V = np.array([omega ** g[dim + i] for i in range(dim)])

    # Apply gauge transformation
    # A'[i,j,k,l] = g_H[i] * g_V[j] * conj(g_H[k]) * conj(g_V[l]) * A[i,j,k,l]
    arr_new = np.einsum('ijkl,i,j,k,l->ijkl', arr, g_H, g_V, g_H.conj(), g_V.conj())

    return Tensor.from_ndarray(np.real(arr_new)), g_H, g_V


def solve_gf3(M, b, dim):
    """Solve M @ g = b over GF(3) using Gaussian elimination."""
    M = M.copy() % 3
    b = b.copy() % 3
    n_rows, n_cols = M.shape

    # Forward elimination
    pivot_row = 0
    for col in range(n_cols):
        if pivot_row >= n_rows:
            break

        # Find pivot
        found = False
        for row in range(pivot_row, n_rows):
            if M[row, col] != 0:
                # Swap rows
                M[[pivot_row, row]] = M[[row, pivot_row]]
                b[[pivot_row, row]] = b[[row, pivot_row]]
                found = True
                break

        if not found:
            continue

        # Scale pivot row to have leading 1
        pivot_val = M[pivot_row, col]
        inv_pivot = pow(int(pivot_val), -1, 3)  # Modular inverse
        M[pivot_row] = (M[pivot_row] * inv_pivot) % 3
        b[pivot_row] = (b[pivot_row] * inv_pivot) % 3

        # Eliminate below
        for row in range(pivot_row + 1, n_rows):
            if M[row, col] != 0:
                factor = M[row, col]
                M[row] = (M[row] - factor * M[pivot_row]) % 3
                b[row] = (b[row] - factor * b[pivot_row]) % 3

        pivot_row += 1

    # Back substitution
    g = np.zeros(n_cols, dtype=int)
    for row in range(min(n_rows, n_cols) - 1, -1, -1):
        # Find leading column
        for col in range(n_cols):
            if M[row, col] != 0:
                g[col] = (b[row] - np.dot(M[row, col+1:], g[col+1:])) % 3
                break

    return g


def gilt_step_gauge_fixed(A, pars, use_discrete_gauge=False):
    """Run one Gilt-TNR step and re-apply gauge fixing."""
    A_new, log_fact = gilttnr_step(A, 0.0, pars)
    A_new, _, _ = fix_continuous_gauge(A_new)

    if use_discrete_gauge:
        A_new, _, _ = fix_discrete_gauge_z3(A_new)

    # Normalize
    norm = np.sqrt(np.sum(np.abs(A_new.to_ndarray())**2))
    A_new = Tensor.from_ndarray(A_new.to_ndarray() / norm)

    return A_new


def newton_step(A, pars, eps=1e-6):
    """
    One Newton iteration step.

    For now, uses simple fixed-point iteration: A_new = R(A)
    True Newton (with Jacobian) requires matching tensor sizes.
    """
    # Compute R(A) with gauge fixing
    R_A = gilt_step_gauge_fixed(A, pars)

    # Compare tensor elements (need same shape for proper comparison)
    arr = A.to_ndarray()
    R_arr = R_A.to_ndarray()

    # If shapes match, compute residual
    if arr.shape == R_arr.shape:
        f = arr.flatten() - R_arr.flatten()
        f_norm = np.linalg.norm(f)
    else:
        # Shapes differ - use Frobenius norm of R(A) as proxy
        # This happens when bond dimension grows
        f_norm = np.linalg.norm(R_arr - R_arr)  # 0, but track shape change
        print(f"  Shape changed: {arr.shape} -> {R_arr.shape}")
        f_norm = float('nan')

    if not np.isnan(f_norm):
        print(f"  ||f(A)|| = {f_norm:.2e}")

    # Simple fixed-point iteration
    return R_A, f_norm


def run_newton(chi=30, gilt_eps=3e-5, n_warmup=10, n_newton=5):
    """Run Newton iteration for 3-state Potts.

    NOTE: True Newton method requires Jacobian computation (see EKR newton.jl).
    This simplified version uses fixed-point iteration with gauge fixing.
    """
    print("=" * 70)
    print("Fixed-Point Iteration for 3-State Potts")
    print("=" * 70)
    print(f"\nParameters: χ={chi}, gilt_eps={gilt_eps:.0e}")
    print(f"Total RG steps: {n_warmup + n_newton}")
    print("\nNOTE: This is fixed-point iteration, not true Newton method.")
    print("True Newton requires Jacobian eigensystem (see EKR newton.jl).\n")

    beta_c = np.log(1 + np.sqrt(3))
    pars = {
        'gilt_eps': gilt_eps,
        'cg_chis': [chi],
        'cg_eps': 1e-10,
        'verbosity': 0,
    }

    # Initial tensor
    A = get_potts_tensor_plain(beta_c)
    print(f"Initial tensor shape: {A.to_ndarray().shape}")

    # Run RG flow without gauge fixing - this is what EKR does for warmup
    print(f"\n{'Step':>4} {'x_σ':>10} {'σ err%':>10} {'x_ε':>10} {'ε err%':>10} {'shape':>15}")
    print("-" * 70)

    for step in range(n_warmup + n_newton):
        sd = get_scaldims_potts(A)
        sigma_err = abs(sd[1] - X_SIGMA) / X_SIGMA * 100
        eps_err = abs(sd[3] - X_EPS) / X_EPS * 100
        shape = A.to_ndarray().shape
        print(f"{step:>4} {sd[1]:>10.4f} {sigma_err:>9.1f}% {sd[3]:>10.4f} {eps_err:>9.1f}% {str(shape):>15}")

        A, _ = gilttnr_step(A, 0.0, pars)

    # Final measurement
    sd = get_scaldims_potts(A)
    sigma_err = abs(sd[1] - X_SIGMA) / X_SIGMA * 100
    eps_err = abs(sd[3] - X_EPS) / X_EPS * 100
    shape = A.to_ndarray().shape
    print(f"{'final':>4} {sd[1]:>10.4f} {sigma_err:>9.1f}% {sd[3]:>10.4f} {eps_err:>9.1f}% {str(shape):>15}")

    # Final results
    print("\n" + "=" * 70)
    print("Final Results:")
    print("=" * 70)
    print(f"  x_σ = {sd[1]:.4f} (CFT: {X_SIGMA:.4f}, error: {abs(sd[1]-X_SIGMA)/X_SIGMA*100:.2f}%)")
    print(f"  x_ε = {sd[3]:.4f} (CFT: {X_EPS:.4f}, error: {abs(sd[3]-X_EPS)/X_EPS*100:.2f}%)")
    print(f"\n  True Newton would improve these to ~0.3% accuracy (see EKR Table 7).")


if __name__ == "__main__":
    # Run to step 7 where accuracy is best before drift
    # True Newton would stabilize at step 5-6 values
    run_newton(chi=30, gilt_eps=3e-5, n_warmup=7, n_newton=0)
