# MODULE: ModelProvider.jl
# USAGE: include("src/ModelProvider.jl") or using EKRNewton
# INPUTS: Model type and parameters (β, etc.)
# OUTPUTS: Initial tensors for TRG, critical parameters, CFT data
# DESCRIPTION: Abstraction layer for Z_N spin models
#
# BASED ON: Tools.jl (Ising-specific functions)
#           GiltTNR2D_Potts.py (Potts tensor construction)
# NEW IN THIS MODULE: Unified interface for Ising, Potts, Clock models

"""
Model provider for Z_N symmetric spin systems.

Provides a unified interface for:
- Initial tensor construction
- Critical point parameters
- CFT scaling dimensions (for verification)
- Symmetry information

Supported models:
- Ising (Z₂): σ ∈ {-1, +1}, β_c = ln(1+√2)/2
- Potts (Z_N): σ ∈ {0, 1, ..., N-1}, Kronecker interaction
- Clock (Z_N): σ ∈ {0, 1, ..., N-1}, cosine interaction
"""

using PyCall

################################################
# Abstract Type Hierarchy
################################################

"""
Abstract base type for all spin models.
"""
abstract type SpinModel end

"""
Abstract type for Z_N symmetric models.
"""
abstract type ZNModel{N} <: SpinModel end

################################################
# Concrete Model Types
################################################

"""
    IsingModel <: ZNModel{2}

2D Ising model on square lattice.
- Symmetry: Z₂
- Action: S = -β Σ_{<ij>} σ_i σ_j, σ ∈ {-1, +1}
- β_c = ln(1+√2)/2 ≈ 0.4407
- CFT: c = 1/2, x_σ = 1/8, x_ε = 1
"""
struct IsingModel <: ZNModel{2} end

"""
    PottsModel{N} <: ZNModel{N}

N-state Potts model on square lattice.
- Symmetry: Z_N (permutation of N states)
- Action: S = -β Σ_{<ij>} δ_{σ_i, σ_j}, σ ∈ {0, 1, ..., N-1}
- β_c = ln(1 + √N)
- CFT data depends on N (N=2: Ising, N=3: c=4/5, N=4: c=1)
"""
struct PottsModel{N} <: ZNModel{N} end

# Convenience constructors
PottsModel(N::Int) = PottsModel{N}()

"""
    ClockModel{N} <: ZNModel{N}

N-state clock model on square lattice.
- Symmetry: Z_N
- Action: S = -β Σ_{<ij>} cos(2π(σ_i - σ_j)/N), σ ∈ {0, 1, ..., N-1}
- For N ≤ 4: single transition, For N ≥ 5: BKT-like behavior
"""
struct ClockModel{N} <: ZNModel{N} end

ClockModel(N::Int) = ClockModel{N}()

################################################
# Symmetry Information
################################################

"""
    symmetry_order(model::SpinModel) -> Int

Return the order N of the Z_N symmetry group.
"""
symmetry_order(::ZNModel{N}) where N = N
symmetry_order(::IsingModel) = 2

"""
    symmetry_group(model::SpinModel) -> Symbol

Return a symbol identifying the symmetry group.
"""
symmetry_group(::ZNModel{N}) where N = Symbol("Z", N)
symmetry_group(::IsingModel) = :Z2

################################################
# Critical Parameters
################################################

"""
    critical_beta(model::SpinModel) -> Float64

Return the exact critical inverse temperature β_c.
"""
critical_beta(::IsingModel) = log(1 + sqrt(2)) / 2

function critical_beta(::PottsModel{N}) where N
    # β_c = ln(1 + √N) for square lattice
    return log(1 + sqrt(N))
end

function critical_beta(::ClockModel{N}) where N
    if N <= 4
        # Has exact solution similar to Potts
        return log(1 + sqrt(N))  # Approximate
    else
        # BKT transitions - no simple formula
        # Return approximate lower transition temperature
        return 1.0  # Placeholder
    end
end

"""
    critical_relT(model::SpinModel) -> Float64

Return relative temperature T/T_c = 1 at criticality.
Used for tensor construction in GiltTNR.
"""
critical_relT(::SpinModel) = 1.0

################################################
# CFT Data
################################################

"""
    central_charge(model::SpinModel) -> Float64

Return the central charge c of the CFT at criticality.
"""
central_charge(::IsingModel) = 0.5  # c = 1/2

function central_charge(::PottsModel{N}) where N
    if N == 2
        return 0.5  # Ising
    elseif N == 3
        return 0.8  # c = 4/5
    elseif N == 4
        return 1.0  # c = 1
    else
        # First-order transition for N > 4
        return NaN
    end
end

central_charge(::ClockModel{N}) where N = N <= 4 ? central_charge(PottsModel{N}()) : NaN

"""
    scaling_dimensions(model::SpinModel) -> NamedTuple

Return key CFT scaling dimensions at criticality.
"""
function scaling_dimensions(::IsingModel)
    return (
        identity = 0.0,
        spin = 1/8,      # x_σ = 1/8
        energy = 1.0,    # x_ε = 1
    )
end

function scaling_dimensions(::PottsModel{N}) where N
    if N == 2
        return scaling_dimensions(IsingModel())
    elseif N == 3
        return (
            identity = 0.0,
            spin = 2/15,     # x_σ = 2/15 ≈ 0.133
            spin_bar = 2/15, # x_σ̄ = 2/15 (degenerate)
            energy = 4/5,    # x_ε = 4/5 = 0.8
        )
    elseif N == 4
        return (
            identity = 0.0,
            spin = 1/8,      # x_σ
            energy = 1/2,    # x_ε
        )
    else
        return (identity = 0.0,)  # No well-defined CFT
    end
end

scaling_dimensions(::ClockModel{N}) where N = scaling_dimensions(PottsModel{N}())

################################################
# Boltzmann Weight Construction
################################################

"""
    boltzmann_weight_matrix(model::SpinModel, β::Real) -> Matrix

Return the Boltzmann weight matrix W[σ, σ'] for nearest-neighbor interaction.
"""
function boltzmann_weight_matrix(::IsingModel, β::Real)
    # W[σ, σ'] = exp(β σ σ'), with σ, σ' ∈ {-1, +1} mapped to {1, 2}
    # W = [exp(β)   exp(-β)]
    #     [exp(-β)  exp(β) ]
    return [exp(β) exp(-β); exp(-β) exp(β)]
end

function boltzmann_weight_matrix(::PottsModel{N}, β::Real) where N
    # W[σ, σ'] = exp(β δ_{σ,σ'}), with σ, σ' ∈ {0, ..., N-1}
    W = ones(N, N)
    for i in 1:N
        W[i, i] = exp(β)
    end
    return W
end

function boltzmann_weight_matrix(::ClockModel{N}, β::Real) where N
    # W[σ, σ'] = exp(β cos(2π(σ-σ')/N))
    W = zeros(N, N)
    for i in 1:N, j in 1:N
        angle = 2π * (i - j) / N
        W[i, j] = exp(β * cos(angle))
    end
    return W
end

################################################
# DFT Matrix for Charge Basis
################################################

"""
    dft_matrix(N::Int) -> Matrix{ComplexF64}

Return the N×N discrete Fourier transform matrix.
F[k, s] = ω^{ks} / √N, where ω = exp(2πi/N)
"""
function dft_matrix(N::Int)
    ω = exp(2π * im / N)
    F = [ω^(k * s) for k in 0:N-1, s in 0:N-1] / sqrt(N)
    return F
end

"""
    dft_matrix_inverse(N::Int) -> Matrix{ComplexF64}

Return the inverse DFT matrix (F†).
"""
dft_matrix_inverse(N::Int) = dft_matrix(N)'

################################################
# Initial Tensor Construction
################################################

"""
    initial_tensor_array(model::SpinModel, β::Real) -> Array{Float64, 4}

Construct the initial 4-leg tensor for TRG in charge basis.

Uses vertex construction (like Ising): T = einsum('ab,bc,cd,da->abcd', W, W, W, W)
followed by Z_N DFT to charge basis.
"""
function initial_tensor_array(model::ZNModel{N}, β::Real) where N
    # Boltzmann weight matrix
    W = boltzmann_weight_matrix(model, β)

    # Vertex construction: T_spin[a,b,c,d] = W[a,b] W[b,c] W[c,d] W[d,a]
    T_spin = zeros(N, N, N, N)
    for a in 1:N, b in 1:N, c in 1:N, d in 1:N
        T_spin[a, b, c, d] = W[a, b] * W[b, c] * W[c, d] * W[d, a]
    end

    # Transform to charge basis via Z_N DFT
    F = dft_matrix(N)
    F_dg = F'

    # T_charge[i,j,k,l] = Σ_{a,b,c,d} F[i,a] F[j,b] F_dg[c,k] F_dg[d,l] T_spin[a,b,c,d]
    T_charge = zeros(ComplexF64, N, N, N, N)
    for i in 1:N, j in 1:N, k in 1:N, l in 1:N
        for a in 1:N, b in 1:N, c in 1:N, d in 1:N
            T_charge[i, j, k, l] += F[i, a] * F[j, b] * F_dg[c, k] * F_dg[d, l] * T_spin[a, b, c, d]
        end
    end

    # Result should be real (for real β)
    T_charge[abs.(T_charge) .< 1e-12] .= 0
    return real.(T_charge)
end

# Ising specialization for efficiency
function initial_tensor_array(::IsingModel, β::Real)
    # Use the standard 2-state construction
    return initial_tensor_array(PottsModel{2}(), β)
end

################################################
# Python Interface
################################################

# Lazy initialization of Python modules
const _py_gilt_modules = Ref{Any}(nothing)

function get_gilt_modules()
    if _py_gilt_modules[] === nothing
        pushfirst!(pyimport("sys")."path", "GiltTNR")
        gilttnr = pyimport("GiltTNR2D")
        tensors = pyimport("tensors")
        _py_gilt_modules[] = (
            gilttnr = gilttnr,
            tensors = tensors,
            TensorZ2 = tensors.TensorZ2,
            Tensor = tensors.Tensor,
        )
    end
    return _py_gilt_modules[]
end

"""
    initial_tensor_py(model::SpinModel, pars::Dict) -> PyObject

Construct initial tensor as Python object for use with GiltTNR.
"""
function initial_tensor_py(model::IsingModel, pars::Dict)
    mods = get_gilt_modules()
    return mods.gilttnr.get_initial_tensor(pars)
end

function initial_tensor_py(model::PottsModel{N}, pars::Dict) where N
    # For Potts, need to use the Potts-specific constructor
    # This will be implemented in GiltTNR2D_ZN.py
    mods = get_gilt_modules()

    # Check if Potts module is available
    if hasproperty(mods.gilttnr, :get_initial_tensor_potts)
        return mods.gilttnr.get_initial_tensor_potts(pars)
    else
        # Fall back to array construction
        β = get(pars, "beta", critical_beta(model))
        arr = initial_tensor_array(model, β)
        return mods.tensors.Tensor.from_ndarray(arr)
    end
end

function initial_tensor_py(model::ClockModel, pars::Dict)
    # Similar to Potts - construct array and wrap as Python Tensor
    mods = get_gilt_modules()
    β = get(pars, "beta", critical_beta(model))
    arr = initial_tensor_array(model, β)
    return mods.tensors.Tensor.from_ndarray(arr)
end

################################################
# Model Description
################################################

"""
    model_name(model::SpinModel) -> String

Return a human-readable name for the model.
"""
model_name(::IsingModel) = "2D Ising"
model_name(::PottsModel{N}) where N = "$N-state Potts"
model_name(::ClockModel{N}) where N = "$N-state Clock"

"""
    model_info(model::SpinModel) -> String

Return a description of the model and its CFT data.
"""
function model_info(model::SpinModel)
    N = symmetry_order(model)
    β_c = critical_beta(model)
    c = central_charge(model)
    dims = scaling_dimensions(model)

    info = """
    Model: $(model_name(model))
    Symmetry: Z_$N
    Critical β: $β_c
    Central charge: $c
    Scaling dimensions: $dims
    """
    return info
end

################################################
# Export
################################################

export SpinModel, ZNModel
export IsingModel, PottsModel, ClockModel
export symmetry_order, symmetry_group
export critical_beta, critical_relT
export central_charge, scaling_dimensions
export boltzmann_weight_matrix, dft_matrix, dft_matrix_inverse
export initial_tensor_array, initial_tensor_py
export model_name, model_info
