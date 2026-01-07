#!/usr/bin/env julia
# MODULE: run_newton_potts3.jl
# USAGE: julia --project=ekrgilttrnr scripts/run_newton_potts3.jl [config]
# INPUTS: Optional config name (quick, default, production)
# OUTPUTS: Scaling dimensions and fixed-point tensor for 3-state Potts
# DESCRIPTION: Full Newton iteration for 3-state Potts model
#
# BASED ON: scripts/test_newton_potts3.jl (infrastructure test)
#           Lab/newton.jl (Ising Newton method)
#           EKR arXiv:2408.10312 (Newton method for TRG)
# NEW IN THIS MODULE: Production-ready Newton for Potts with data saving

using Dates

println("=" ^ 70)
println("Newton Iteration for 3-State Potts")
println("=" ^ 70)
println("Started: $(now())")

# Get project root directory
const PROJECT_ROOT = dirname(@__DIR__)

# Parse config from command line
config_name = length(ARGS) > 0 ? ARGS[1] : "quick"
println("Config: $config_name")

# Configuration presets
# IMPORTANT: Potts CFT fixed point drifts after ~7 steps with GILT-TNR
# Need to stop warmup at optimal accuracy point (step 3-4 for x_σ, step 7 for x_ε)
configs = Dict(
    "quick" => Dict(
        "chi" => 30,        # Higher chi for better fixed point
        "gilt_eps" => 3e-5,
        "n_warmup" => 4,    # Stop at optimal x_σ accuracy (~0.2%)
        "n_newton" => 5,
        "jacobian_rank" => 6,
    ),
    "default" => Dict(
        "chi" => 30,
        "gilt_eps" => 3e-5,
        "n_warmup" => 5,    # Balance between x_σ and x_ε accuracy
        "n_newton" => 10,
        "jacobian_rank" => 9,
    ),
    "production" => Dict(
        "chi" => 40,
        "gilt_eps" => 3e-5,
        "n_warmup" => 6,    # Higher chi allows slightly more warmup
        "n_newton" => 20,
        "jacobian_rank" => 12,
    ),
)

if !haskey(configs, config_name)
    error("Unknown config: $config_name. Available: $(keys(configs))")
end
cfg = configs[config_name]

# Include the modules
println("\nLoading modules...")
include(joinpath(PROJECT_ROOT, "src", "EchelonFormGFN.jl"))
include(joinpath(PROJECT_ROOT, "src", "ZNTensor.jl"))
include(joinpath(PROJECT_ROOT, "src", "ContinuousGauge.jl"))
include(joinpath(PROJECT_ROOT, "src", "DiscreteGaugeZN.jl"))
include(joinpath(PROJECT_ROOT, "src", "ModelProvider.jl"))
include(joinpath(PROJECT_ROOT, "src", "NewtonZN.jl"))

using LinearAlgebra
using PyCall

# Setup Python path
pushfirst!(pyimport("sys")."path", joinpath(PROJECT_ROOT, "GiltTNR"))

# CFT reference values for 3-state Potts
const X_SIGMA = 2/15  # ≈ 0.1333
const X_EPS = 4/5     # = 0.8

println("\n" * "=" ^ 70)
println("Model Parameters")
println("=" ^ 70)

model = PottsModel(3)
println("Model: $(model_name(model))")
println("Symmetry: Z_$(symmetry_order(model))")
println("Critical β: $(critical_beta(model))")

dims = scaling_dimensions(model)
println("CFT scaling dimensions:")
println("  x_σ = $(dims.spin) ($(round(dims.spin, digits=4)))")
println("  x_ε = $(dims.energy)")

println("\n" * "=" ^ 70)
println("Configuration")
println("=" ^ 70)
for (k, v) in cfg
    println("  $k = $v")
end

# Run Newton iteration
println("\n" * "=" ^ 70)
println("Running Newton Iteration")
println("=" ^ 70)

config = NewtonConfig(
    model;
    chi = cfg["chi"],
    gilt_eps = cfg["gilt_eps"],
    jacobian_rank = cfg["jacobian_rank"],
    verbosity = 1,
)

# Get initial tensor as Python object
# Use pycall to prevent auto-conversion
tensors_mod = pyimport("tensors")
β = critical_beta(model)
A_arr = initial_tensor_array(model, β)
A_init = pycall(tensors_mod.Tensor.from_ndarray, PyObject, A_arr)
println("Initial tensor type: $(typeof(A_init))")

# Run Newton
try
    result = newton_iteration(
        A_init, config;
        n_warmup = cfg["n_warmup"],
        n_newton = cfg["n_newton"],
    )

    # Extract results
    eigenvalues = result["eigenvalues"]
    residuals = result["residuals"]
    A_final = result["A_final"]

    println("\n" * "=" ^ 70)
    println("Results")
    println("=" ^ 70)

    println("\nJacobian eigenvalues (relevant for scaling dimensions):")
    for (i, λ) in enumerate(eigenvalues)
        x = -log(abs(λ)) / (2π)
        println("  λ_$i = $(round(abs(λ), digits=6)), x_$i = $(round(x, digits=4))")
    end

    println("\nFinal residuals:")
    println("  ||δA|| at step 1: $(residuals[1])")
    println("  ||δA|| at final: $(residuals[end])")

    # Save results
    data_dir = joinpath(PROJECT_ROOT, "data", "potts3_newton")
    mkpath(data_dir)

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename = joinpath(data_dir, "potts3_$(config_name)_$timestamp.csv")

    open(filename, "w") do f
        println(f, "# 3-State Potts Newton Results")
        println(f, "# Config: $config_name")
        println(f, "# chi=$(cfg["chi"]), gilt_eps=$(cfg["gilt_eps"])")
        println(f, "# n_warmup=$(cfg["n_warmup"]), n_newton=$(cfg["n_newton"])")
        println(f, "# CFT: x_sigma=$(X_SIGMA), x_epsilon=$(X_EPS)")
        println(f, "# ")
        println(f, "# Jacobian eigenvalues:")
        println(f, "index,eigenvalue_abs,eigenvalue_real,eigenvalue_imag,scaling_dim")
        for (i, λ) in enumerate(eigenvalues)
            x = -log(abs(λ)) / (2π)
            println(f, "$i,$(abs(λ)),$(real(λ)),$(imag(λ)),$x")
        end
        println(f, "# ")
        println(f, "# Residuals:")
        println(f, "# step,residual")
        for (i, r) in enumerate(residuals)
            println(f, "# $i,$r")
        end
    end

    println("\nResults saved to: $filename")

catch e
    println("\nNewton iteration failed:")
    println(e)
    println("\nThis may be due to:")
    println("  - Insufficient warmup (try more n_warmup)")
    println("  - Bond dimension too low (try higher chi)")
    println("  - Numerical instability in Jacobian computation")
end

println("\n" * "=" ^ 70)
println("Completed: $(now())")
println("=" ^ 70)
