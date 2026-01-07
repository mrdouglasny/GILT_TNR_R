#!/usr/bin/env julia
# MODULE: test_shape_preservation.jl
# USAGE: julia --project=ekrgilttrnr scripts/test_shape_preservation.jl
# INPUTS: None
# OUTPUTS: Shape preservation test results
# DESCRIPTION: Test whether disabling GILT preserves tensor shapes during RG

println("=" ^ 70)
println("Shape Preservation Test for Potts Newton")
println("=" ^ 70)

const PROJECT_ROOT = dirname(@__DIR__)

# Include modules
include(joinpath(PROJECT_ROOT, "src", "EchelonFormGFN.jl"))
include(joinpath(PROJECT_ROOT, "src", "ZNTensor.jl"))
include(joinpath(PROJECT_ROOT, "src", "ContinuousGauge.jl"))
include(joinpath(PROJECT_ROOT, "src", "DiscreteGaugeZN.jl"))
include(joinpath(PROJECT_ROOT, "src", "ModelProvider.jl"))

using LinearAlgebra
using PyCall

# Setup Python
pushfirst!(pyimport("sys")."path", joinpath(PROJECT_ROOT, "GiltTNR"))
gilttnr = pyimport("GiltTNR2D")
tensors_mod = pyimport("tensors")

# Create Potts initial tensor
model = PottsModel(3)
β = critical_beta(model)
A_arr = initial_tensor_array(model, β)
A_init = pycall(tensors_mod.Tensor.from_ndarray, PyObject, A_arr)

println("\nInitial tensor shape: $(size(A_arr))")

# Test different configurations
configs = [
    ("GILT + TRG adaptive", Dict("gilt_eps" => 3e-5, "cg_chis" => collect(1:16), "cg_eps" => 1e-10, "verbosity" => 0)),
    ("GILT + TRG fixed chi", Dict("gilt_eps" => 3e-5, "cg_chis" => [16], "cg_eps" => 1e-10, "verbosity" => 0)),
    ("TRG only (gilt_eps=0)", Dict("gilt_eps" => 0.0, "cg_chis" => [16], "cg_eps" => 1e-10, "verbosity" => 0)),
]

for (name, pars) in configs
    println("\n" * "-" ^ 50)
    println("Config: $name")
    println("-" ^ 50)

    # Start from initial tensor
    A = A_init

    # Run a few warmup steps
    shapes = [size(A.to_ndarray())]
    for step in 1:5
        result = pycall(gilttnr.gilttnr_step, PyObject, A, 0.0, pars)
        A = get(result, PyObject, 0)  # Python tuples are 0-indexed
        push!(shapes, size(A.to_ndarray()))
    end

    println("Shapes after each step:")
    for (i, s) in enumerate(shapes)
        println("  Step $(i-1): $s")
    end

    # Check if shape is stable (after first few warmup steps when shape grows)
    final_shape = shapes[end]
    # Shape should be stable after step 3 (when it reaches chi)
    stable = all(s == final_shape for s in shapes[4:end])
    println("Shape stable after step 3: $stable")

    # Test perturbation - this is what happens during Jacobian computation
    if stable
        println("\nTesting perturbation stability...")
        A_arr_final = A.to_ndarray()
        δ = 1e-4 * randn(size(A_arr_final)...)
        A_perturbed = pycall(tensors_mod.Tensor.from_ndarray, PyObject, A_arr_final + δ)

        result = pycall(gilttnr.gilttnr_step, PyObject, A_perturbed, 0.0, pars)
        A_perturbed_evolved = get(result, PyObject, 0)
        perturbed_shape = size(A_perturbed_evolved.to_ndarray())

        println("  Original shape after warmup: $final_shape")
        println("  Perturbed tensor evolved shape: $perturbed_shape")
        println("  Shape preserved under perturbation: $(perturbed_shape == final_shape)")
    end
end

println("\n" * "=" ^ 70)
println("Test completed")
println("=" ^ 70)
