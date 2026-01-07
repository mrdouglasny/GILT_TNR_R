# Rotations, Negative Eigenvalues, and Newton Method in Tensor Network Renormalization Group

**Authors:** Nikolay Ebel, Tom Kennedy, Slava Rychkov
**arXiv:** [2408.10312](https://arxiv.org/abs/2408.10312)
**Published:** Physical Review X, Vol. 15, No. 3, Article 031047 (2025)

## Summary

**Context and Main Results:**
This paper develops a Newton method to find fixed point tensors in tensor network RG, achieving ~10⁻⁹ accuracy on the fixed point equation. The key innovation is composing the RG map with a π/2 rotation to eliminate marginal deformations that would otherwise create continuous families of fixed points. Applied to 2D Ising and 3-state Potts using Gilt-TNR at χ=30.

**Key Results:**
- Newton method converges to ||R∘(A*) - A*|| ~ 10⁻¹³
- Ising scaling dimensions: 0.01-0.02% error
- Potts scaling dimensions: 0.3-3% error (requires larger ε_gilt)
- Jacobian eigenvalues give CFT data directly

## Technical Details

### Newton Method Algorithm

Fixed point equation with rotation:
```
R∘ = Γ_{π/2} ∘ R
```
where Γ_{π/2} is 90° rotation. Newton iteration:
```
A_{m+1} = A_m - [J_≈(A_m)]^{-1} · f(A_m)
f(A) = A - R∘(A)
J(A) = I - ∇R∘(A)
```

### Jacobian Eigenvalue to Scaling Dimension

```
λ = (i)^{ℓ_O} · b^{2 - Δ_O}
```
where:
- ℓ_O = operator spin
- Δ_O = scaling dimension
- b = 2 (lattice rescaling)
- Factor of i from π/2 rotation

### Gauge Fixing (Two-Stage)

**Continuous stage:**
- Construct environments from tensor copies
- Diagonalize with orthogonal matrices G_v, G_h
- Apply to vertical/horizontal legs

**Discrete stage:**
- Solve linear system in modular arithmetic (Z₂ for Ising, GF(3) for Potts)
- Use largest-magnitude elements to fix remaining sign/phase freedom

### Parameters

| Model | χ | ε_gilt | Notes |
|-------|---|--------|-------|
| Ising | 30 | 6×10⁻⁶ | Standard |
| Potts | 30 | 3×10⁻⁵ | Larger due to Z₃ sectors |

### 3-State Potts Implementation

Tensor in charge basis with eigenvectors |0⟩, |1⟩, |2⟩ of Z₃ generator η:
```
η|q⟩ = e^{2πiq/3}|q⟩,  q ∈ {0, 1, 2}
```

**Important:** They use **plain tensors** expressed in charge basis, NOT block-diagonal symmetric tensors. The RG map preserves S₃ symmetry by construction.

The larger ε_gilt for Potts reflects:
1. More charge sectors → larger intermediate bond dimensions
2. Need to balance disentangling vs coarse-graining errors

### Scaling Dimension Results

**Ising (χ=30):**
| Operator | λ measured | CFT exact | Error |
|----------|------------|-----------|-------|
| ε | 1.9996 | 2.0 | 0.02% |
| σ | 3.6684 | 3.668 | 0.01% |
| T, T̄ | -1.00 | -1.0 | <0.1% |

**3-State Potts (χ=30):**
| Operator | λ measured | CFT exact | Error |
|----------|------------|-----------|-------|
| ε | 2.2897 | 2.2974 | 0.3% |
| Φ, Φ̄ | 1.1456i | 1.1487i | 0.3% |
| T, T̄ | -0.998 | -1.0 | 0.4-3.1% |

## Relevance to Project

**Critical for potts3/ekrgilttrnr subprojects:**

1. **Parameters confirmed:** Our tests with χ=30, ε_gilt=3×10⁻⁵ for Potts match their recommendations
2. **TensorZ3 incompatibility explained:** They use plain tensors in charge basis, which is why our TensorZ3 attempts failed
3. **Newton method approach:** Can adapt their gauge fixing for our newton.jl implementation
4. **Discrete gauge fixing:** Need GF(3) linear algebra (already in TensorKitPotts.jl)

**Key insight:** Don't use block-diagonal symmetric tensors for Gilt-TNR. Use plain tensors and let the symmetry be preserved by construction.

## BibTeX

```bibtex
@article{Ebel2025newton,
  title={Rotations, Negative Eigenvalues, and Newton Method in Tensor Network Renormalization Group},
  author={Ebel, Nikolay and Kennedy, Tom and Rychkov, Slava},
  journal={Physical Review X},
  volume={15},
  number={3},
  pages={031047},
  year={2025},
  eprint={2408.10312},
  archivePrefix={arXiv},
  primaryClass={cond-mat.stat-mech}
}
```
