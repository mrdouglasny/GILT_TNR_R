#!/usr/bin/env python3
"""
Reproduce EKR (arXiv:2408.10312) 3-state Potts results.

Their parameters from Appendix F:
- χ = 30
- gilt_eps = 3×10⁻⁵
- Plain tensors (NOT TensorZ3)
- Tensor in charge basis (Z₃ DFT of vertex construction)

Their results (Table 7):
- ε (energy): λ = 2.2897 vs CFT 2.2974 (0.3% error)
  → x_ε = 2 - log_2(2.2897) = 2 - 1.195 = 0.805
- Φ, Φ̄ (spin): λ = 1.1456i vs CFT 1.1487i (0.3% error)
  → x_σ = 2 - log_2(1.1456) = 2 - 0.196 = 1.804... wait that's wrong

Let me recalculate. Their convention:
  λ = b^{2-Δ} where b=2, Δ = scaling dimension
  So Δ = 2 - log_2(|λ|)

For spin: |λ| = 1.1456, Δ = 2 - log_2(1.1456) = 2 - 0.196 = 1.804 ???

That doesn't match x_σ = 2/15 = 0.133...

Oh wait - they report JACOBIAN eigenvalues, not transfer matrix eigenvalues.
The Jacobian eigenvalue formula includes rotation: λ_J = i^{ℓ} b^{2-Δ}

For scaling dimensions from transfer matrix (what we compute):
  x = -log(|λ_TM|/|λ_0|) / π

Let's just run the flow and compare.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'GiltTNR'))

import numpy as np
from tensors import Tensor
from ncon import ncon
from GiltTNR2D import gilttnr_step
from GiltTNR2D_Potts import get_scaldims_potts

# CFT values for 3-state Potts (c=4/5)
X_SIGMA = 2/15  # 0.1333
X_EPS = 4/5     # 0.8

def get_potts_tensor_plain(beta):
    """
    Build plain Potts tensor in charge basis (like EKR).

    Uses vertex construction + Z₃ DFT, but returns plain Tensor.
    """
    q = 3

    # Boltzmann weight matrix
    W = np.ones((q, q))
    np.fill_diagonal(W, np.exp(beta))

    # Vertex construction: T[a,b,c,d] = W[a,b] W[b,c] W[c,d] W[d,a]
    T_spin = np.einsum('ab,bc,cd,da->abcd', W, W, W, W)

    # Z₃ DFT to charge basis
    omega = np.exp(2j * np.pi / 3)
    F = np.array([
        [1, 1, 1],
        [1, omega, omega**2],
        [1, omega**2, omega]
    ]) / np.sqrt(3)
    F_dg = F.T.conj()

    # Transform: T' = F⊗F⊗F†⊗F† T
    T_charge = ncon(
        (T_spin, F, F, F_dg, F_dg),
        ([1, 2, 3, 4], [-1, 1], [-2, 2], [3, -3], [4, -4])
    )

    # Clean up numerical noise
    T_charge[np.abs(T_charge) < 1e-12] = 0

    # Return as plain Tensor (real part - imaginary should be ~0)
    return Tensor.from_ndarray(np.real(T_charge))

def run_flow(chi, gilt_eps, n_steps):
    """Run Gilt-TNR flow and track scaling dimensions."""

    # Critical point
    beta_c = np.log(1 + np.sqrt(3))

    pars = {
        'gilt_eps': gilt_eps,
        'cg_chis': [chi],
        'cg_eps': 1e-10,
        'verbosity': 0,
    }

    A = get_potts_tensor_plain(beta_c)
    log_fact = 0.0

    results = []
    for step in range(n_steps + 1):
        sd = get_scaldims_potts(A)
        results.append({
            'step': step,
            'x_1': sd[1],
            'x_2': sd[2],
            'x_3': sd[3] if len(sd) > 3 else np.nan,
        })

        if step < n_steps:
            A, log_fact = gilttnr_step(A, log_fact, pars)

    return results

def main():
    print("=" * 70)
    print("Reproducing EKR (arXiv:2408.10312) 3-State Potts Results")
    print("=" * 70)
    print(f"\nCFT targets: x_σ = {X_SIGMA:.4f} (2/15), x_ε = {X_EPS:.4f} (4/5)")
    print("\nEKR parameters: χ=30, gilt_eps=3×10⁻⁵")
    print("-" * 70)

    # EKR parameters
    chi = 30
    gilt_eps = 3e-5
    n_steps = 7

    results = run_flow(chi, gilt_eps, n_steps)

    print(f"\n{'Step':>4} {'x_1 (σ)':>12} {'x_2 (σ̄)':>12} {'x_3 (ε)':>12} {'σ err%':>10} {'ε err%':>10}")
    print("-" * 70)

    for r in results:
        sigma_err = abs(r['x_1'] - X_SIGMA) / X_SIGMA * 100
        eps_err = abs(r['x_3'] - X_EPS) / X_EPS * 100 if not np.isnan(r['x_3']) else np.nan
        print(f"{r['step']:>4} {r['x_1']:>12.4f} {r['x_2']:>12.4f} {r['x_3']:>12.4f} {sigma_err:>9.1f}% {eps_err:>9.1f}%")

    # Final results
    final = results[-1]
    print("\n" + "=" * 70)
    print("Final Results (after step", n_steps, "):")
    print("=" * 70)
    print(f"  x_σ = {final['x_1']:.4f} (CFT: {X_SIGMA:.4f}, error: {abs(final['x_1']-X_SIGMA)/X_SIGMA*100:.1f}%)")
    print(f"  x_ε = {final['x_3']:.4f} (CFT: {X_EPS:.4f}, error: {abs(final['x_3']-X_EPS)/X_EPS*100:.1f}%)")

    # Compare with different chi values
    print("\n" + "=" * 70)
    print("Chi Comparison (at step 5):")
    print("=" * 70)

    for test_chi in [16, 20, 24, 30]:
        results = run_flow(test_chi, gilt_eps, 5)
        r = results[-1]  # step 5
        sigma_err = abs(r['x_1'] - X_SIGMA) / X_SIGMA * 100
        eps_err = abs(r['x_3'] - X_EPS) / X_EPS * 100
        print(f"  χ={test_chi:>2}: x_σ={r['x_1']:.4f} ({sigma_err:>5.1f}%), x_ε={r['x_3']:.4f} ({eps_err:>5.1f}%)")

if __name__ == "__main__":
    main()
