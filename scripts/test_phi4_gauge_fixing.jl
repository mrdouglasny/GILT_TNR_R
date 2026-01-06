# MODULE: test_phi4_gauge_fixing.jl
# USAGE: julia --project=ekrgilttrnr ekrgilttrnr/scripts/test_phi4_gauge_fixing.jl
# DESCRIPTION: Diagnose gauge fixing issues for φ⁴ tensors
#
# This script checks:
# 1. Whether continuous gauge fixing produces diagonal environments
# 2. Whether discrete gauge fixing finds enough independent constraints

using LinearAlgebra

# Load tools
include("../src/Tools.jl")
include("../src/GaugeFixing.jl")
include("../src/Phi4Tools.jl")

println("=" ^ 60)
println("GAUGE FIXING DIAGNOSTICS FOR φ⁴")
println("=" ^ 60)

# Parameters - use higher chi to keep larger tensors
chi = 30
gilt_pars = Dict(
    "gilt_eps" => 6e-6,
    "cg_chis" => collect(1:chi),
    "cg_eps" => 1e-10,
    "verbosity" => 0,
    "rotate" => true,
)

########################################
# Test: φ⁴ tensor
########################################
println("\n--- φ⁴ MODEL GAUGE FIXING TEST ---")

phi4_pars = Dict(
    "mu_sq" => -3.0,  # Near critical
    "lam" => 1.0,
    "kappa" => 1.0,
    "K" => 32,
    "D" => 16,
    "symmetry_tensors" => true
)

A_phi4 = initial_tensor_phi4(phi4_pars)
println("Initial φ⁴ tensor shape: $(A_phi4.shape)")

# Run more RG steps to get closer to fixed point
println("Running 15 RG steps...")
for i in 1:15
    global A_phi4
    A_phi4, _ = py"gilttnr_step"(A_phi4, 0.0, gilt_pars)
    if i % 5 == 0
        println("  Step $i: shape=$(A_phi4.shape)")
    end
end
println("After RG, tensor shape: $(A_phi4.shape)")

# Apply continuous gauge fixing
A_phi4_cg, H, V, SH, SV = fix_continuous_gauge(A_phi4)

# Check environment is diagonal
env_h = environment_for_horizontal_gauge([A_phi4_cg, A_phi4_cg.conj()]).to_ndarray()
env_v = environment_for_vertical_gauge([A_phi4_cg, A_phi4_cg.conj()]).to_ndarray()
off_diag_h = norm(env_h - diagm(diag(env_h)))
off_diag_v = norm(env_v - diagm(diag(env_v)))
println("Continuous gauge - off-diagonal norm (H): $off_diag_h")
println("Continuous gauge - off-diagonal norm (V): $off_diag_v")

# Check discrete gauge fixing
println("\nAnalyzing tensor structure for discrete gauge...")
list_of_elements = list_of_allowed_elements(A_phi4_cg; tol=1e-7)
sort!(list_of_elements, by = x -> -abs(x[2]))
println("Number of non-zero off-diagonal elements (|val| > 1e-7): $(length(list_of_elements))")
println("Top 10 elements by magnitude:")
for (i, el) in enumerate(list_of_elements[1:min(10, length(list_of_elements))])
    println("  $i: index=$(el[1]), value=$(el[2])")
end

# Apply discrete gauge fixing
A_phi4_dg, accepted_elements_phi4, HZ2, VZ2 = fix_discrete_gauge(A_phi4_cg)
println("\nDiscrete gauge - found $(length(accepted_elements_phi4)) independent constraints")
dimH = sum(A_phi4_cg.shape[1, :])
dimV = sum(A_phi4_cg.shape[2, :])
expected_dofs = dimH + dimV - 3
println("Expected DOFs: $expected_dofs (dimH=$dimH, dimV=$dimV)")

if length(accepted_elements_phi4) < expected_dofs
    deficit = expected_dofs - length(accepted_elements_phi4)
    println("\n⚠️  GAUGE FIXING INCOMPLETE: Missing $deficit DOFs")
    println("   This will cause Newton method to fail!")
end

# Analyze why elements are not independent
println("\n--- Analyzing constraint independence ---")
dimH = sum(A_phi4_cg.shape[1, :])
dimV = sum(A_phi4_cg.shape[2, :])

# Build the constraint matrix for the first few elements
println("Constraint vectors for top elements:")
for (i, el) in enumerate(list_of_elements[1:min(10, length(list_of_elements))])
    idx = el[1]
    raw = zeros(Bool, dimH + dimV)
    raw[idx[1]] = !raw[idx[1]]
    raw[idx[3]] = !raw[idx[3]]
    raw[dimH + idx[2]] = !raw[dimH + idx[2]]
    raw[dimH + idx[4]] = !raw[dimH + idx[4]]
    println("  $i: $(Int.(raw))")
end

println("\n" * "=" ^ 60)
println("DIAGNOSIS COMPLETE")
println("=" ^ 60)
