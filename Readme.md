# ekrgilttrnr - EKR Gilt-TNR Implementation

## Overview

This subproject contains the Gilt-TNR implementation from Ebel, Kennedy, and Rychkov's papers:
- **arXiv:2408.10312** - "Rotations, Negative Eigenvalues, and Newton Method in Tensor Network Renormalization Group" (Phys. Rev. X 15, 031023, 2025)
- Transfer Matrix and Lattice Dilatation Operator paper

**Purpose:** Reproduce their eigenvalue extraction results for the 2D Ising model and extend to other models (φ⁴ theory).

**Source:** https://github.com/ebelnikola/GILT_TNR_R

## Supported Models

### 2D Ising Model
The original implementation. Critical exponents extracted via Newton method.

### 2D φ⁴ Theory (NEW)
Lattice scalar field theory with Z₂ symmetry. Same universality class as Ising (c=1/2 CFT).

Action: S = Σ_x [½(∂φ)² + ½μ²φ² + ¼λφ⁴]

## Key Results

Comparison of scaling dimensions (exponents) $\Delta$ obtained from deriving the transfer matrix spectrum.

| Operator | Ising (Exact) | φ⁴ (This Work) |
|----------|---------------|----------------|
| Magnetization ($\sigma$) | 0.125 | 0.152 |
| Energy ($\epsilon$) | 1.000 | 0.998 |

Values for φ⁴ were computed at RG step 5 with $\chi=32$ (see `docs/phi4_results.pdf`).

## Directory Structure

```
ekrgilttrnr/
├── src/
│   ├── GiltTNR/              # Python Gilt-TNR library
│   │   ├── GiltTNR2D_Ising_benchmarks.py  # Ising tensors
│   │   └── GiltTNR2D_Phi4.py              # φ⁴ tensors (NEW)
│   ├── Tools.jl              # Ising utilities
│   ├── Phi4Tools.jl          # φ⁴ utilities (NEW)
│   ├── GaugeFixing.jl        # Gauge fixing routines
│   ├── KrylovTechnical.jl    # Z₂ invariant tensors
│   └── NumDifferentiation.jl # Finite differences
├── scripts/
│   ├── critical_temperature.jl  # Ising critical T search
│   ├── eigensystem.jl           # Ising eigenvalues
│   ├── newton.jl                # Ising Newton method
│   ├── phi4_critical_mu_sq.jl   # φ⁴ critical μ² search (NEW)
│   ├── phi4_eigensystem.jl      # φ⁴ eigenvalues (NEW)
│   └── phi4_newton.jl           # φ⁴ Newton method (NEW)
├── docs/
│   └── newton_method_guide.md   # Detailed methodology guide (NEW)
├── Project.toml
└── VERIFICATION.md
```

## Installation

### 1. Julia Dependencies

```bash
cd ekrgilttrnr
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### 2. NumPy Compatibility Fix (Partial)

Fixed `np.float_` → `np.float64` for NumPy 2.x, but deeper issues remain:

```bash
cd ekrgilttrnr/src/GiltTNR
sed -i 's/np\.float_/np.float64/g' tensors/abeliantensor.py tensors/symmetrytensors.py plots/plotFError.py
```

### 3. Jupyter Notebook (Recommended)

The easiest way to use this code is via the Jupyter notebook:

```bash
cd ekrgilttrnr

# Install Julia 1.10.4 if not already present
juliaup add 1.10.4

# Install IJulia kernel for Julia 1.10.4
# This does NOT change your global default Julia version
julia +1.10.4 --project=. -e 'using Pkg; Pkg.add("IJulia"); using IJulia; IJulia.installkernel("Julia-1.10")'

# Open the notebook in VS Code
# Use Newton_method_fixed.ipynb (modified for our directory structure)
```

**To run the notebook:**
1. Open `Newton_method_fixed.ipynb` in VS Code
2. Select the "Julia 1.10" kernel when prompted
3. Run cells sequentially (Shift+Enter) or "Run All"

**Expected results (chi=30, default parameters):**
- Critical temperature: relT ≈ 1.0000110042840245
- Eigenvalues:
  - σ (magnetization): λ = 3.6684 (CFT: 3.668)
  - ε (energy): λ = 1.9996 (CFT: 2.0)
  - T (stress): λ = 1.0015 (CFT: 1.0)
  - T̄ (stress): λ = 0.9980 (CFT: 1.0)

### 4. Known Issues (Direct Script Usage)

⚠️ **The direct Julia scripts may have compatibility issues** with our environment setup. The Jupyter notebook approach is recommended as it matches the authors' original workflow.

**For comparison with `genmodel/scripts/linearized_rg.jl`:** Use published eigenvalue results from arXiv:2408.10312 (λ_ε=1.9996, λ_T=1.0015/0.9980, λ_σ=3.6684).

## Usage

### Customizing Parameters

The key parameters are set in **cell 8** of `Newton_method_fixed.ipynb`:

```julia
gilt_eps = 6e-6      # GILT truncation threshold
chi = 30             # Bond dimension
cg_eps = 1e-10       # Coarse-graining precision
Jratio = 1.0         # Coupling ratio (1.0 for standard Ising)
```

**To explore different parameters:**
1. Modify cell 8 in the notebook
2. Rerun from cell 8 onwards

**Common variations:**
- `chi = 20`: Faster, less accurate (paper used chi=16,20,24,30)
- `chi = 40`: Slower, more accurate (computationally expensive)
- `gilt_eps = 1e-5`: Faster convergence, lower precision
- `gilt_eps = 1e-7`: Slower, higher precision

**Note:** Changing `chi` or `gilt_eps` will find a different critical temperature and yield different eigenvalues.

### Compute Eigenvalues (Main Task)

**Recommended approach:** Use the Jupyter notebook (see Installation §3 above)

**Alternative (direct script):**
```bash
cd ekrgilttrnr
julia --project=. scripts/eigensystem.jl
```

Output: `eigensystems/*.data` (serialized Julia data)

## φ⁴ Theory Usage

The φ⁴ model has been adapted to use the same Newton method machinery as the Ising model.

### Quick Start for φ⁴

```bash
cd ekrgilttrnr

# 1. Find critical μ² (takes ~20-30 min)
julia --project scripts/phi4_critical_mu_sq.jl --chi 30

# 2. Compute eigenvalues at criticality
julia --project scripts/phi4_eigensystem.jl --chi 30

# 3. Run Newton method for high-precision fixed point
julia --project --threads 20 scripts/phi4_newton.jl --chi 30 --eigensystem_size_for_jacobian 54
```

### φ⁴ Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mu_sq` | auto | Mass squared (uses critical value if 0) |
| `--lam` | 1.0 | Quartic coupling λ |
| `--kappa` | 1.0 | Kinetic coupling κ |
| `--K` | 32 | Quadrature points |
| `--D` | 16 | Initial bond dimension |

**Critical point estimate:** For λ=1, κ=1, the critical μ² ≈ -0.09 (from λ/|μ²_c| ≈ 10.9).

### Expected Results for φ⁴

Since φ⁴ is in the same universality class as 2D Ising:
- Eigenvalues should match Ising: λ_σ ≈ 3.668, λ_ε ≈ 2.0, λ_T ≈ 1.0
- This provides a non-trivial validation of universality

## Documentation

See `docs/newton_method_guide.md` for:
- Detailed explanation of the Newton method algorithm
- How gauge fixing works
- How to adapt to other models (XY, Potts, etc.)
- Troubleshooting guide

## References

- arXiv:2408.10312 - Main paper
- refs/2408.10312/summary.md - Summary
- genmodel/docs/ekr_eigenvalue_comparison.md - Comparison
- https://github.com/ebelnikola/GILT_TNR_R
