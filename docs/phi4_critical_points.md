# φ⁴ Critical Points Reference

## Critical μ² Values

The critical mass parameter μ²_c depends on the kinetic coupling κ. Based on numerical scans:

| κ     | μ²_c (approx) | Notes |
|-------|---------------|-------|
| 0.3   | -1.275        | χ=15, gilt_eps=6e-6 |
| 1.0   | ~-0.1         | (needs verification) |

### Scaling Relationship

Empirically, μ²_c scales roughly as:
```
μ²_c ≈ -4κ
```

This means:
- For κ = 0.3: μ²_c ≈ -1.2 (actual: -1.275)
- For κ = 1.0: μ²_c ≈ -4.0 (or closer to 0 depending on other params)

## Saved Critical Values

Critical values are saved to `critical_temperatures/` with naming:
```
phi4_rotate={rotate}_{chi}_{gilt_eps}_{cg_eps}_lam={lam}_kappa={kappa}_K={K}_D={D}_tol={tol}.data
```

The eigensystem script looks up saved values automatically when `mu_sq = 0` in config.

## Phase Behavior

At the critical point:
- Tensor bond dimension stays at χ (doesn't collapse)
- RG flow is slow (many steps to converge)
- Off-critical: tensor collapses to trivial [1,1,1,1] or [2,2,2,2]

## Diagnostic Signs

**Too far from critical (μ² too high or too low):**
- Tensor collapses to small dimension after ~10 RG steps
- Eigenvalues are 0 or undefined

**Near critical but not exact:**
- Tensor stays at full χ for ~10 steps then collapses
- Eigenvalues are large/wrong (e.g., ~800 instead of ~3.6)

**At critical point:**
- Tensor dimension stable for many RG steps
- Eigenvalues match 2D Ising CFT: λ_σ ≈ 3.668, λ_ε ≈ 2.0, λ_T ≈ 1.0

## Running Critical Point Search

```bash
# For κ = 0.3 (our standard)
julia --project=ekrgilttrnr ekrgilttrnr/scripts/phi4_critical_mu_sq.jl \
  --chi 15 --kappa 0.3 \
  --mu_sq_low -1.5 --mu_sq_high -1.0 \
  --search_tol 1e-6

# For κ = 1.0
julia --project=ekrgilttrnr ekrgilttrnr/scripts/phi4_critical_mu_sq.jl \
  --chi 15 --kappa 1.0 \
  --mu_sq_low -0.5 --mu_sq_high 0.0 \
  --search_tol 1e-6
```

## Gauge Fixing for φ⁴ Tensors

### Methods Comparison (validated 2026-01-02)

| Method | Environment Diagonality | Jacobian ||df|| | Status |
|--------|------------------------|----------|--------|
| **environment** | ~10⁻¹⁶ ✅ | 7.95 | Recommended |
| svd | ~0.019 ❌ | 9.16 | Does not fix gauge |
| none | ~0.019 ❌ | 9.16 | No gauge fixing |

**Key finding**: The `svd` method (simple normalization + phase fix) does NOT actually fix the continuous gauge freedom. Only the `environment` method properly diagonalizes the transfer matrix environments.

### Environment-Based Gauge Fixing

The environment method computes:
1. Vertical environment: contract A with A† over legs 0,2,3 leaving leg 1
2. Diagonalize this environment with eigendecomposition
3. Apply gauge transformation to diagonalize vertical
4. Repeat for horizontal environment

This is the same approach used for 2D Ising (`fix_continuous_gauge` in GaugeFixing.jl).

### Fixed Point Quality

**CRITICAL**: Before computing eigenvalues, check the fixed point residual:
```
||RG(A) - A|| should be << 1
```

At μ² = -1.325 (current):
- ||RG(A) - A|| ≈ 0.97 after 10 steps (BAD - not at fixed point)
- Tensor collapses after 15+ steps

This explains the spurious large eigenvalues (~2000). The Jacobian captures the flow toward collapse, not the scaling dimensions.

### Requirements for Valid Eigenvalues

1. **Precise critical μ²**: Need tolerance better than 1e-4
2. **Fixed point convergence**: ||RG(A) - A|| < 1e-2 ideally
3. **Stable bond dimension**: Tensor should stay at χ for 20+ steps
4. **Environment gauge fixing**: Use `method=:environment` not `:svd`

## History

- 2026-01-02: Found μ²_c ≈ -1.275 for κ=0.3, χ=15, lam=1.0
- 2026-01-02: Validated gauge fixing methods; environment method works, svd does not
- 2026-01-02: Identified fixed point quality issue (||RG(A)-A|| too large)
