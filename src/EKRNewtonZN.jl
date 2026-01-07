# MODULE: EKRNewtonZN.jl
# USAGE: include("src/EKRNewtonZN.jl")
# INPUTS: Model parameters
# OUTPUTS: Newton-converged fixed point tensors
# DESCRIPTION: Main module for Z_N Newton method infrastructure
#
# BASED ON: EKR arXiv:2408.10312, Lab/newton.jl
# NEW IN THIS MODULE: Unified infrastructure for arbitrary Z_N models

"""
EKRNewtonZN - Generalized Newton Method for Z_N Symmetric Tensor Networks

This module provides Newton iteration for finding fixed points of the
Gilt-TNR renormalization group for Z_N symmetric spin models.

Supported models:
- Ising (N=2): σ ∈ {-1, +1}
- Potts (N=3,4,...): σ ∈ {0, 1, ..., N-1}, Kronecker interaction
- Clock (N states): σ ∈ {0, 1, ..., N-1}, cosine interaction

Key components:
- ZNTensor{N}: Julia wrapper for Z_N symmetric tensors (KrylovKit compatible)
- EchelonFormGFN: Linear algebra over finite fields GF(N)
- ContinuousGauge: Environment diagonalization (model-independent)
- DiscreteGaugeZN: GF(N) phase fixing (requires Z_N symmetry)
- ModelProvider: Abstraction for spin models
- NewtonZN: Newton iteration algorithm

Usage:
    include("src/EKRNewtonZN.jl")

    # Run Newton for 3-state Potts
    result = run_newton_potts(3; chi=30, n_warmup=20, n_newton=30)

    # Access results
    A_fixed = result["A_final"]
    eigenvalues = result["eigenvalues"]
"""

module EKRNewtonZN

using LinearAlgebra
using PyCall
using KrylovKit

# Include all components
include("EchelonFormGFN.jl")
include("ZNTensor.jl")
include("ContinuousGauge.jl")
include("DiscreteGaugeZN.jl")
include("ModelProvider.jl")
include("NewtonZN.jl")

# Re-export key types and functions
export SpinModel, ZNModel, IsingModel, PottsModel, ClockModel
export ZNTensor, Z2Tensor, Z3Tensor, Z4Tensor
export symmetry_order, critical_beta, scaling_dimensions

export fix_continuous_gauge
export fix_discrete_gauge_zn, fix_discrete_gauge_z2

export NewtonConfig
export newton_iteration, newton_step
export run_newton_ising, run_newton_potts

export py_to_zn, zn_to_py
export initial_tensor_array, initial_tensor_py

export solve_gfn, echelon_form_gfn!, rank_gfn

end # module
