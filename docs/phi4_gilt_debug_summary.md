# φ⁴ GILT-TNR Debugging Summary

**Date:** 2026-01-05
**Status:** Identified root cause of eigenvalue extraction failure

## Problem

Eigenvalue extraction for φ⁴ returns garbage values (~9600) instead of expected CFT values (λ_σ ≈ 3.67, λ_ε ≈ 2.0, λ_T ≈ 1.0).

## Root Cause

The φ⁴ tensor **always collapses** to a trivial fixed point after ~10-20 RG steps:

1. **Symmetric phase (μ² > μ²_c):** Tensor collapses to [1,1,1,1]
2. **Broken phase (μ² < μ²_c):** Tensor collapses to [2,2,2,2] (doubly-degenerate)

Neither fixed point is the **critical fixed point** needed for eigenvalue extraction.

## Critical Point Location

For parameters (λ=1.0, κ=0.3, K=32, D=16, χ=30):

```
μ²_c ≈ -1.32796 ± 0.00002
```

- μ² = -1.32795: symmetric phase (collapses after 25 steps)
- μ² = -1.32798: broken phase (collapses after 24 steps)

**Note:** Previous searches used μ² ≈ -1.27 to -1.325, which are all in the symmetric phase!

## Comparison with Ising

| Aspect | Ising (β_c) | φ⁴ (μ²_c) |
|--------|-------------|-----------|
| λ₂ trajectory | 0.71 → 0.94 (stable) | 0.71 → 0.84 → collapses |
| λ₃ trajectory | 0.28 → 0.50 (stable) | 0.24 → 0.60 → 0 |
| Dimension | Stays at 15-16 | Collapses to 1 or 2 |
| Fixed point | Stable critical | Unstable critical |

## Why Ising Works but φ⁴ Fails

1. **Ising at β_c:** The critical fixed point is an **attractor** with a basin of attraction. The Ising tensor flows toward and stays near the critical point.

2. **φ⁴ at μ²_c:** The critical fixed point is a **saddle point**. Any deviation from the exact critical manifold causes the tensor to flow away - either to the symmetric (trivial) or broken (doubly-degenerate) fixed point.

## Implications

- Eigenvalue extraction requires the tensor to be **at the fixed point**: ||RG(A) - A|| ≈ 0
- For φ⁴ with GILT-TNR, this condition is never satisfied for more than a few RG steps
- The finite-difference Jacobian computation sees a collapsing tensor, producing garbage

## Potential Solutions

### 1. Newton's Method for Fixed Point
Instead of running many RG steps and hoping to reach the fixed point, use Newton iteration to solve RG(A) = A directly. The EKR code has `newton.jl` for this.

### 2. Higher Bond Dimension / Quadrature
Current: K=32, D=16, χ=30. Try K=64, D=32, χ=60. Higher resolution might stabilize the critical trajectory.

### 3. Different Disentangler
GILT may not effectively remove CDL entanglement for continuous field theories. Consider:
- Loop-TNR with explicit disentangler optimization
- MERA-style disentangling

### 4. Use genmodel/TNRKit Instead
The genmodel implementation with TNRKit.jl (BTRG/HOTRG) successfully extracted φ⁴ critical exponents (ν=0.98±0.02). This approach works because it computes observables (free energy, correlation length) rather than linearized RG eigenvalues.

## Data Files

- Test scripts: `ekrgilttrnr/scripts/compare_ising_phi4.jl`
- Trajectory debug: `ekrgilttrnr/scripts/debug_phi4_linearized.jl`
- Critical point search: in session (not saved as permanent script)

## Recommended Next Steps

1. **Short term:** Use genmodel/TNRKit approach for φ⁴ critical exponents
2. **Medium term:** Implement Newton fixed-point finder for GILT-TNR
3. **Long term:** Investigate alternative disentanglers for continuous fields
