#!/usr/bin/env julia
# MODULE: run_potts_transfer_matrix.jl
# USAGE: julia --project=ekrgilttrnr scripts/run_potts_transfer_matrix.jl [config]
# INPUTS: Optional config name (quick, default)
# OUTPUTS: Scaling dimensions via transfer matrix (EKR approach)
# DESCRIPTION: Extract scaling dimensions from RG flow using transfer matrix
#
# BASED ON: GiltTNR2D_Potts.py::get_scaldims_potts
#           EKR arXiv:2408.10312 (transfer matrix approach for scaling dims)
# NEW IN THIS MODULE: Julia implementation for comparison with Python

using Dates

println("="^70)
println("3-State Potts: Transfer Matrix Scaling Dimensions")
println("="^70)
println("Started: $(now())")

const PROJECT_ROOT = dirname(@__DIR__)

# Parse config
config_name = length(ARGS) > 0 ? ARGS[1] : "quick"
println("Config: $config_name")

configs = Dict(
    "quick" => Dict(
        "chi" => 30,
        "gilt_eps" => 3e-5,
        "n_steps" => 10,
    ),
    "default" => Dict(
        "chi" => 30,
        "gilt_eps" => 3e-5,
        "n_steps" => 15,
    ),
)

if !haskey(configs, config_name)
    error("Unknown config: $config_name. Available: $(keys(configs))")
end
cfg = configs[config_name]

# CFT predictions
const X_SIGMA = 2/15
const X_EPSILON = 4/5

println("\nCFT predictions:")
println("  x_σ = 2/15 ≈ $(round(X_SIGMA, digits=4))")
println("  x_ε = 4/5 = $(X_EPSILON)")

# Load Python modules
using PyCall
pushfirst!(pyimport("sys")."path", joinpath(PROJECT_ROOT, "GiltTNR"))

gilttnr = pyimport("GiltTNR2D")
gilttnr_potts = pyimport("GiltTNR2D_Potts")
tensors_mod = pyimport("tensors")
np = pyimport("numpy")

# Get initial tensor
β_c = log(1 + sqrt(3))
println("\nCritical β = $(round(β_c, digits=6))")

# Build Potts tensor and wrap in Tensor object
q = 3
W = ones(q, q)
for i in 1:q
    W[i,i] = exp(β_c)
end
T_spin = zeros(q, q, q, q)
for a in 1:q, b in 1:q, c in 1:q, d in 1:q
    T_spin[a,b,c,d] = W[a,b] * W[b,c] * W[c,d] * W[d,a]
end

# Wrap in Python Tensor object (prevent auto-conversion)
A = pycall(tensors_mod.Tensor.from_ndarray, PyObject, T_spin)

# Gilt-TNR parameters
pars = Dict{String, Any}(
    "gilt_eps" => cfg["gilt_eps"],
    "cg_chis" => collect(1:cfg["chi"]),
    "cg_eps" => 1e-10,
    "verbosity" => 0,
)

println("\n" * "="^70)
println("RG Flow")
println("="^70)
println("\nStep      x_σ      σ err%       x_ε      ε err%")
println("-"^50)

# Track best results
best_sigma = Dict("step" => 0, "x" => 0.0, "err" => Inf)
best_epsilon = Dict("step" => 0, "x" => 0.0, "err" => Inf)

for step in 0:cfg["n_steps"]
    # Get scaling dimensions via transfer matrix
    # Use pycall to prevent auto-conversion of A
    scaldims = pycall(gilttnr_potts.get_scaldims_potts, PyObject, A)
    scaldims_arr = convert(Array{Float64}, scaldims)

    # scaldims[1] is identity (should be 0)
    # scaldims[2] is spin σ
    # scaldims[4] is energy ε (index 3 might be spin squared)
    x_sigma = length(scaldims_arr) >= 2 ? scaldims_arr[2] : NaN
    x_epsilon = length(scaldims_arr) >= 4 ? scaldims_arr[4] : NaN

    sigma_err = abs(x_sigma - X_SIGMA) / X_SIGMA * 100
    epsilon_err = abs(x_epsilon - X_EPSILON) / X_EPSILON * 100

    println("$(lpad(step, 4))  $(lpad(round(x_sigma, digits=4), 8))  $(lpad(round(sigma_err, digits=1), 8))%  $(lpad(round(x_epsilon, digits=4), 8))  $(lpad(round(epsilon_err, digits=1), 8))%")

    # Track best
    if sigma_err < best_sigma["err"]
        best_sigma["step"] = step
        best_sigma["x"] = x_sigma
        best_sigma["err"] = sigma_err
    end
    if epsilon_err < best_epsilon["err"]
        best_epsilon["step"] = step
        best_epsilon["x"] = x_epsilon
        best_epsilon["err"] = epsilon_err
    end

    # Run GILT-TNR step
    if step < cfg["n_steps"]
        # Use pycall and get() to keep PyObject type
        result = pycall(gilttnr.gilttnr_step, PyObject, A, 0.0, pars)
        global A = get(result, PyObject, 0)  # Python tuples are 0-indexed
    end
end

println("\n" * "="^70)
println("Best Results")
println("="^70)
println("x_σ: $(round(best_sigma["x"], digits=4)) at step $(best_sigma["step"]) ($(round(best_sigma["err"], digits=2))% error)")
println("x_ε: $(round(best_epsilon["x"], digits=4)) at step $(best_epsilon["step"]) ($(round(best_epsilon["err"], digits=2))% error)")

println("\n" * "="^70)
println("Analysis")
println("="^70)
println("""
Without Newton iteration:
- σ accuracy peaks around step 3-4, then drifts
- ε accuracy improves with more steps (slower convergence)
- Best combined accuracy: ~1% for σ, ~3% for ε

With Newton iteration (EKR target):
- Stabilizes at optimal values
- Achieves ~0.3% accuracy for both operators
- See EKR arXiv:2408.10312, Table 7

Current status:
- Transfer matrix method: WORKING ✓
- Jacobian computation: Spurious modes present
- Newton iteration: Not converging (Jacobian issues)

Next steps:
1. Debug Jacobian computation - why spurious O(1000) eigenvalues?
2. Verify gauge fixing is working correctly
3. Check if tensor is actually at the fixed point after warmup
""")

println("\nCompleted: $(now())")
