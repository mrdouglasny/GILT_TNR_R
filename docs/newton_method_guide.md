# Newton Method for Finding Fixed Point Tensors in TRG

## Overview

The `Newton_method_fixed.ipynb` notebook implements the Newton method approach from Ebel, Kennedy, and Rychkov's paper "Rotations, Negative Eigenvalues, and Newton Method in Tensor Network Renormalization Group" (Phys. Rev. X 15, 031023, 2025; arXiv:2408.10312).

**Goal:** Find the fixed point tensor A* of the Gilt-TNR renormalization group map with high precision (~10^-9), then extract the Jacobian eigenvalues to determine critical exponents.

---

## 1. Algorithm Pipeline

The notebook follows this pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Find Critical Temperature                                      │
│ ─────────────────────────────────────────────────────────────────────── │
│ Binary search to locate T_c where trajectory doesn't flow to either    │
│ high-T or low-T fixed point                                             │
└────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Generate Initial Approximation                                 │
│ ─────────────────────────────────────────────────────────────────────── │
│ Run 23 RG steps at critical T to get approximate fixed point tensor    │
└────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Fix Gauge Freedom                                              │
│ ─────────────────────────────────────────────────────────────────────── │
│ - Continuous gauge: diagonalize transfer matrix environments           │
│ - Discrete gauge: fix signs of tensor elements                         │
└────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: Compute Jacobian Eigensystem                                   │
│ ─────────────────────────────────────────────────────────────────────── │
│ Use Krylov methods (Arnoldi iteration) to find dominant eigenvalues    │
│ and eigenvectors of the linearized RG map                              │
└────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 5: Newton Iteration                                               │
│ ─────────────────────────────────────────────────────────────────────── │
│ A_{m+1} = A_m - (I - J)^{-1} (A_m - R(A_m))                            │
│ where R is the RG map and J is the (approximate) Jacobian              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Key Components Explained

### 2.1 The Gilt-TNR Algorithm

**Location:** `src/GiltTNR/` (Python) called via PyCall

The Gilt-TNR (Graph-Independent Local Truncation TNR) algorithm combines:
- **TRG (Tensor Renormalization Group):** Coarse-graining via SVD
- **GILT:** Removes entanglement from tensor legs before truncation

The RG step is called as:
```julia
A_new, log_fact, errs = py"gilttnr_step"(A, log_fact, gilt_pars)
```

**Parameters:**
- `gilt_eps = 6e-6` — Threshold for GILT truncation
- `chi = 30` — Maximum bond dimension
- `cg_eps = 1e-10` — Threshold for TRG truncation
- `rotate = true/false` — Whether to rotate 90° after each step

### 2.2 Critical Temperature Search

**Location:** `src/Tools.jl:396-458` (function `perform_search`)

The algorithm uses binary search:
1. Start with `relT_low = 1.0` (low-T phase) and `relT_high = 1.01` (high-T phase)
2. Run RG trajectory for midpoint temperature
3. Check second eigenvalue of tensor:
   - If → 1: low-T phase (ordered)
   - If → 0: high-T phase (disordered)
4. Bisect until `gap < 10^-10`

Result for Ising: `relT_c ≈ 1.000013...` (slightly above T_c due to finite χ)

### 2.3 Gauge Fixing

**Location:** `src/GaugeFixing.jl`

Tensors have gauge freedom under:
- **Continuous gauge:** Orthogonal transformations on bond indices
- **Discrete gauge:** Sign flips on bond indices

**Continuous Gauge Fixing:**
```julia
A, H, V, SH, SV = fix_continuous_gauge(A)
```
This diagonalizes the transfer matrix environments:
- Compute environment E_h = Tr_{vert}(A ⊗ A*)
- Diagonalize E_h → eigenvalues SH, eigenvectors H
- Transform A with H on horizontal legs
- Repeat for vertical legs

**Discrete Gauge Fixing:**
```julia
A, accepted_elements, H, V = fix_discrete_gauge(A; tol=1e-7)
```
Fixes sign ambiguity by requiring certain tensor elements to be positive.

### 2.4 Numerical Differentiation

**Location:** `src/NumDifferentiation.jl`

Computes directional derivatives via finite differences:
```julia
df(f, x, v; stp=1e-4, order=2) = Σ_i c_i * f(x + offset_i * stp * v) / stp
```

Orders available:
- `order=2`: 2nd order accurate (coefficients: -1/2, +1/2)
- `order=3`: 4th order accurate (coefficients: 1/12, -2/3, +2/3, -1/12)
- `order=4`: 6th order accurate

### 2.5 Krylov Eigensolver

**Location:** Uses `KrylovKit.jl` package

The Jacobian is never formed explicitly. Instead:
```julia
dgilt(δA) = df(x -> gilt(x, gilt_pars), A_crit, δA; stp=1e-4, order=2)
```

Then use Arnoldi iteration:
```julia
eigensystem = eigsolve(dgilt, initial_vector, num_eigenvalues, :LM;
                       krylovdim=num_eigenvalues+20)
```

This finds the `num_eigenvalues` largest-magnitude eigenvalues.

### 2.6 Newton's Method

**Location:** `scripts/newton.jl:155-273`

The fixed point equation is `A* = R(A*)` where R is the RG map.

Newton's method solves `(I - J)(A* - A_m) = A_m - R(A_m)`:

```julia
function newton_function(A)
    x_minus_f = A - gilt(A, gilt_pars)          # A - R(A)
    correction = ImJ_inv(x_minus_f)             # (I - J)^{-1} * (A - R(A))
    return A - correction                        # Newton update
end
```

**Key insight:** The Jacobian J is approximated using only `s` eigenvectors:
- Project to subspace Vs spanned by eigenvectors
- Invert (I - J) exactly in Vs
- Identity in orthogonal complement (assumes small eigenvalues ≈ 0)

```julia
function ImJ_inv(δA)
    δA_in_Vs = project_to_Vs(δA)       # Project to eigenvector subspace
    δA_out_of_Vs = δA - δA_in_Vs       # Orthogonal complement

    # Invert (I-J) in subspace using precomputed matrix
    ImJ_inv_δA_in_Vs = Σ_{i,j} ImJ_inv_matrix[i,j] * e_i * ⟨e_j, δA⟩

    return ImJ_inv_δA_in_Vs + δA_out_of_Vs  # Identity on complement
end
```

### 2.7 The Rotation Trick

**Why rotate?** The stress tensor operators T, T̄ have scaling dimension Δ=2, giving eigenvalues λ=1 (marginal). This creates a continuous family of fixed points.

**Solution:** Rotate the lattice 90° after each RG step:
- Scaling dimension 2 → eigenvalue 2^(2-2) = 1
- With rotation: eigenvalue becomes -1 (sign flip)
- This isolates the fixed point, enabling Newton convergence

```julia
gilt_pars = Dict(
    "rotate" => true,  # Enable 90° rotation
    ...
)
```

---

## 3. Data Structures

### 3.1 Z2Tensor (Julia)

**Location:** `src/KrylovTechnical.jl:55-66`

```julia
struct Z2Tensor
    sects::Dict{NTuple{4, Int64}, Array}  # Tensor blocks by Z2 quantum numbers
    shape::Matrix{Int64}                   # Block dimensions
    qhape::Matrix{Int64}                   # Quantum number assignments
    dirs::Vector{Int64}                    # Index directions (in/out)
end
```

This represents Z2-symmetric tensors efficiently by storing only non-zero blocks.

### 3.2 Python TensorZ2

The actual computation uses Python tensors (`tensors.TensorZ2` from GiltTNR library). Conversion:
- `py_to_ju(A)` — Python → Julia
- `ju_to_py(A)` — Julia → Python

---

## 4. Expected Results

### 4.1 For 2D Ising at χ=30

**Non-rotating Gilt-TNR eigenvalues:**
| Operator | CFT Δ | CFT λ | Computed λ |
|----------|-------|-------|------------|
| σ (magnetization) | 1/8 | 3.668 | 3.6684 |
| ε (energy) | 1 | 2.0 | 1.9996 |
| T (stress) | 2 | 1.0 | 1.0015 |
| T̄ (stress) | 2 | 1.0 | 0.9980 |

**Rotating Gilt-TNR eigenvalues:**
- Stress tensor eigenvalues flip sign: λ_T ≈ -1.0010, λ_T̄ ≈ -0.9982

### 4.2 Newton Convergence

With s=54 eigenvectors for Jacobian approximation:
- Converges in ~15 iterations
- Final step size: ~10^-10
- Fixed point accuracy: ~10^-9

---

## 5. Adapting to Other Models

### 5.1 Changing the Initial Tensor

The initial tensor for the 2D Ising model is created in:
```python
# In GiltTNR2D_essentials.py
A_0 = get_initial_tensor_aniso({"relT": relT, "Jratio": Jratio, "symmetry_tensors": True})
```

**For a new model:**
1. Create a function that constructs the initial tensor A from physical parameters
2. The tensor should have the same index structure: A[left, up, right, down]
3. If the model has Z2 symmetry, use TensorZ2 format
4. If no Z2 symmetry, use regular numpy arrays (and modify Z2Tensor handling)

### 5.2 Key Modifications Needed

#### Step 1: Initial Tensor Construction

Create a Python function in `GiltTNR/`:
```python
def get_initial_tensor_XY(pars):
    """Construct initial tensor for XY model"""
    beta = pars["beta"]
    chi = pars.get("chi", 30)

    # Construct Boltzmann weights
    # Return as TensorZ2 if model has Z2 symmetry
    # Otherwise return as Tensor
```

#### Step 2: Modify Julia Interface

In `src/Tools.jl`, add:
```julia
function initial_tensor_XY(pars)
    A_0 = py"get_initial_tensor_XY"(pars)
    return A_0
end
```

#### Step 3: Phase Detection

Modify `phase()` function in `src/Tools.jl:396-407` for your model:
```julia
function phase_XY(A, gilt_pars; max_steps=40, tol=1e-4)
    for i in 1:max_steps
        A, _ = py"gilttnr_step"(A, 0.0, gilt_pars)
        # Check for your model's phase indicators
        # e.g., check vortex density, magnetization pattern
    end
end
```

#### Step 4: Critical Parameter Search

Modify `perform_search()` to search over your model's control parameter:
```julia
function perform_search_XY(beta_low, beta_high, gilt_pars; search_tol=1e-5)
    # Binary search over beta instead of relT
end
```

### 5.3 Models Without Z2 Symmetry

For models without Z2 symmetry (e.g., 3-state Potts, XY):
1. Use regular numpy arrays instead of TensorZ2
2. Modify `KrylovTechnical.jl` to work with plain arrays
3. The gauge fixing still works (it's the continuous U(1) gauge from transfer matrices)

### 5.4 Example: Adapting for XY Model

```julia
# 1. Define XY initial tensor
function initial_tensor_XY(pars)
    beta = pars["beta"]
    K = pars.get("K", 64)  # Discretization
    # XY model initial tensor from Bessel function expansion
    return py"get_XY_tensor"(beta, K)
end

# 2. XY phase detection (look for KT transition)
function phase_XY(A, gilt_pars; max_steps=40)
    for i in 1:max_steps
        A, _ = py"gilttnr_step"(A, 0.0, gilt_pars)
        # XY model: check correlation decay rate
        # High-T: exponential decay
        # Low-T: power-law decay (quasi-long-range order)
    end
end

# 3. Rest of pipeline (gauge fixing, Newton) works unchanged
```

### 5.5 Handling Different Symmetries

| Model | Symmetry | Tensor Type | Gauge Fixing |
|-------|----------|-------------|--------------|
| Ising | Z2 | TensorZ2 | Standard |
| 3-Potts | Z3 | TensorZ3 | Modify for Z3 |
| XY | U(1) | Regular Tensor | Standard |
| Clock | Z_N | TensorZN | Modify for Z_N |

---

## 6. Computational Considerations

### 6.1 Memory Requirements

- Tensor A: O(χ^4) elements = O(810,000) for χ=30
- Jacobian approximation: O(s × χ^4) where s = number of eigenvectors
- Full Jacobian (never formed): O(χ^8) elements

### 6.2 Computational Cost

- Each Gilt-TNR step: O(χ^6) operations
- Jacobian-vector product (numerical diff): 4-6 Gilt-TNR steps
- Arnoldi iteration for s eigenvalues: O(s × krylovdim) Jacobian-vector products
- Newton iteration: O(s^2) per step (for projection operations)

### 6.3 Parallelization

- The Gilt-TNR step is internally parallelized (BLAS)
- Jacobian-vector products can be parallelized across directions
- Use `--threads N` when running Julia

---

## 7. File Outputs

| Directory | Contents |
|-----------|----------|
| `critical_temperatures/` | Serialized critical temperature data |
| `trajectories/` | RG trajectories (tensor history) |
| `eigensystems/` | Computed eigenvalues and eigenvectors |
| `newton/` | Newton iteration results |
| `trajectory_plots/` | Visualization PDFs |
| `export/` | Publication-quality figures |

---

## 8. Troubleshooting

### Issue: "Critical relT was not found"
- Run `critical_temperature.jl` first to generate the data

### Issue: Gauge fixing warnings
- Normal at trajectory start (tensor shape changes)
- Warnings about "below threshold" elements are acceptable

### Issue: Newton doesn't converge
- Increase `eigensystem_size_for_jacobian` (default: 54)
- Check that you're using rotating algorithm (`rotate=true`)
- Ensure initial approximation is close enough (increase `number_of_initial_steps`)

### Issue: Eigenvalues don't match CFT
- Increase χ (bond dimension)
- Decrease `gilt_eps` (higher precision)
- Check gauge fixing is working correctly

---

## 9. References

1. **arXiv:2408.10312** — Main paper for this implementation
2. **arXiv:2506.03247** — Follow-up paper on computer-assisted proofs
3. **Levin-Nave 2006** — Original TRG algorithm
4. **Evenbly-Vidal 2015** — TNR (Tensor Network Renormalization)
5. **Hauru et al. 2018** — GILT algorithm

---

## 10. Quick Start

```bash
cd ekrgilttrnr

# 1. Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# 2. Find critical temperature (takes ~20 min)
julia --project scripts/critical_temperature.jl

# 3. Compute eigenvalues (takes ~10 min)
julia --project scripts/eigensystem.jl

# 4. Run Newton method (takes ~15 min)
julia --project --threads 20 scripts/newton.jl --eigensystem_size_for_jacobian 54

# Or use the Jupyter notebook for interactive exploration
jupyter notebook Newton_method_fixed.ipynb
```
