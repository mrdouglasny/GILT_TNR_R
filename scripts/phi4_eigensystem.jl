# MODULE: phi4_eigensystem.jl
# USAGE: julia +1.10.4 --project=ekrgilttrnr ekrgilttrnr/scripts/phi4_eigensystem.jl <config>
# INPUTS: ekrgilttrnr/configs/phi4_eigensystem/<config>.toml
# OUTPUTS: ekrgilttrnr/data/phi4_eigensystem/<config>.csv
# DESCRIPTION: Compute RG Jacobian eigenvalues for φ⁴ model at criticality
# LATEX_DOC: ekrgilttrnr/docs/newton_method_guide.md
#
# BASED ON: EKR GILT-TNR (https://github.com/ebelnikola/GILT_TNR_R, arXiv:2408.10312),
#           ekrgilttrnr/scripts/eigensystem.jl (Ising version)
# NEW IN THIS SCRIPT: Computes RG Jacobian eigenvalues for φ⁴ to verify universality.
#                     Expected to match 2D Ising CFT (c=1/2): λ_σ≈3.668, λ_ε≈2.0, λ_T≈1.0.
#
# RUNTIME: ~5 min (quick), ~30 min (default χ=30). Scales as O(χ⁶).
#
# Quick test run (< 5 min):
#   julia --project=ekrgilttrnr ekrgilttrnr/scripts/phi4_eigensystem.jl quick
#
# Production run (~30 min):
#   julia --project=ekrgilttrnr ekrgilttrnr/scripts/phi4_eigensystem.jl default

################################################
# section: EXPERIMENT INIT
################################################

using Serialization
using LinearAlgebra
using KrylovKit

# Load ProjectUtils for config handling (from rg/src/julia/)
include("../../src/julia/ProjectUtils.jl")
using .ProjectUtils

# Load tools (from ekrgilttrnr/src/)
include("../src/Tools.jl")
include("../src/Phi4Tools.jl")
include("../src/GaugeFixing.jl")
include("../src/Phi4GaugeFixing.jl")
include("../src/KrylovTechnical.jl")
include("../src/NumDifferentiation.jl")
using .NumDifferentiation

# Setup with config (loads TOML, creates paths)
# base_dir="ekrgilttrnr" since scripts run from rg/ on cluster
config_name, config, paths = setup_with_config("phi4_eigensystem", ARGS; base_dir="ekrgilttrnr")

# Extract parameters with defaults
chi = get(config, "chi", 30)
gilt_eps = get(config, "gilt_eps", 6e-6)
cg_eps = get(config, "cg_eps", 1e-10)
rotate = get(config, "rotate", false)
lam = get(config, "lam", 1.0)
kappa = get(config, "kappa", 0.3)
K = get(config, "K", 32)
D = get(config, "D", 16)
mu_sq = get(config, "mu_sq", 0.0)  # 0 means lookup from saved critical values
number_of_initial_steps = get(config, "number_of_initial_steps", 10)
ord = get(config, "ord", 2)
stp = get(config, "stp", 1e-4)
N = get(config, "N", 10)
krylovdim = get(config, "krylovdim", 25)
verbosity = get(config, "verbosity", 1)
search_tol = get(config, "search_tol", 1e-6)

# Build gilt_pars early for lookup
gilt_pars_lookup = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => cg_eps,
    "verbosity" => 0,
    "rotate" => rotate,
)

# If mu_sq is 0 or not set, look up saved critical value
if mu_sq == 0.0
    result = lookup_critical_mu_sq(lam, kappa, K, D, gilt_pars_lookup, search_tol)
    if result !== nothing
        _, _, mu_sq = result
        println("Using saved critical μ² = $mu_sq")
    else
        error("No saved critical μ² found. Run phi4_critical_mu_sq.jl first with parameters: lam=$lam, kappa=$kappa, K=$K, D=$D")
    end
end

################################################
# section: SETUP
################################################

gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => cg_eps,
    "verbosity" => verbosity > 1 ? 2 : 0,
    "rotate" => rotate,
)

phi4_pars = Dict(
    "mu_sq" => mu_sq,
    "lam" => lam,
    "kappa" => kappa,
    "K" => K,
    "D" => D,
    "symmetry_tensors" => true
)

# Log parameters
log_msg = """
==================================================
φ⁴ Jacobian Eigenvalue Computation
Config: $config_name
==================================================
Parameters:
  μ² = $mu_sq, λ = $lam, κ = $kappa
  χ = $chi, gilt_eps = $gilt_eps
  K = $K, D = $D
  number_of_initial_steps = $number_of_initial_steps
"""
log_info(log_msg, paths["log"])
if verbosity > 0
    println(log_msg)
end

################################################
# section: RG FLOW TO FIXED POINT
################################################

# Create initial tensor
A0 = initial_tensor_phi4(phi4_pars)

# Run RG to get near fixed point
trajectory_result = phi4_trajectory(phi4_pars, number_of_initial_steps, gilt_pars)
A_crit = trajectory_result["A"][end]

# Gauge fix using environment-based method (validated in test_gauge_fixing.jl)
# Environment method properly diagonalizes transfer matrix environments,
# which is critical for linearized RG to remove spurious gauge directions
A_crit, gauge_info = fix_gauge_phi4(A_crit; method=:environment)
if verbosity > 0
    println("Gauge fixed with method: $(gauge_info["method"])")
end

A_crit_JU = py_to_ju(A_crit)

# Debugging
println("Norm of A_crit: ", norm(A_crit_JU))
# A_next = gilt_phi4(A_crit_JU) # gilt_phi4 is not defined yet


################################################
# section: EIGENVALUES
################################################

# Define RG step function (applies RG + gauge fixes + normalizes)
# CRITICAL: Gauge fixing MUST be inside this function for finite difference to work
# Using environment method (diagonalizes environments, removes gauge directions)
function gilt_phi4(A_ju)
    A_py = ju_to_py(A_ju)
    getitem = pyimport("operator").getitem
    result = pycall(py"gilttnr_step", PyObject, A_py, 0.0, gilt_pars)
    A_out = pycall(getitem, PyObject, result, 0)
    # Apply environment-based gauge fixing (validated to properly fix gauge)
    A_out, _ = fix_gauge_phi4(A_out; method=:environment)
    return py_to_ju(A_out)
end

# Debugging
println("Norm of A_crit: ", norm(A_crit_JU))
A_next = gilt_phi4(A_crit_JU)
println("Norm of gilt_phi4(A_crit): ", norm(A_next))
diff = A_next - A_crit_JU
println("Norm of difference: ", norm(diff))

# Helper to project tensor onto a fixed structure (template)
function project_to_structure(A::Z2Tensor, template::Z2Tensor)
    new_sects = Dict{NTuple{4, Int64}, Array}()
    for (k, v_temp) in template.sects
        if haskey(A.sects, k)
            v_A = A.sects[k]
            if size(v_A) == size(v_temp)
                new_sects[k] = v_A
            else
                # Resize v_A to match v_temp
                new_v = zeros(eltype(v_A), size(v_temp))
                common_size = min.(size(v_A), size(v_temp))
                ranges = [1:s for s in common_size]
                new_v[ranges...] = v_A[ranges...]
                new_sects[k] = new_v
            end
        else
            # Missing sector, treat as zero
            # println("Warning: Dropping sector $k in project_to_structure")
            new_sects[k] = zeros(eltype(template.sects[k]), size(template.sects[k]))
        end
    end
    return Z2Tensor(new_sects, template.shape, template.qhape, template.dirs)
end

# Define linearized RG map using numerical differentiation
function linearized_RG(dA)
    out = NumDifferentiation.df(gilt_phi4, A_crit_JU, dA; stp=stp, order=ord)
    println("Linearized RG: input norm $(norm(dA)), output norm $(norm(out))")
    return project_to_structure(out, A_crit_JU)
end

# Debug linearized_RG
function random_like(A)
    new_sects = Dict{NTuple{4, Int64}, Array}()
    for (k, v) in A.sects
        new_sects[k] = rand(eltype(v), size(v))
    end
    return Z2Tensor(new_sects, A.shape, A.qhape, A.dirs)
end

rand_vec = random_like(A_crit_JU)
rand_vec = (1.0 / norm(rand_vec)) * rand_vec
res_vec = linearized_RG(rand_vec)
println("Norm of random vector: ", norm(rand_vec))
println("Norm of linearized_RG(rand_vec): ", norm(res_vec))


# Compute eigenvalues using Krylov
if verbosity > 0
    println("\nComputing eigenvalues...")
end

# Use a random starting vector instead of A_crit_JU to avoid getting stuck in the trivial direction
x0 = random_like(A_crit_JU)
vals, vecs, info = eigsolve(linearized_RG, x0, N, :LM;
                            krylovdim=krylovdim, verbosity=2) # Increased verbosity

println("Eigsolve info: ", info)


# Format results
eigenvalue_log = "\nEigenvalues:\n"
for i in 1:min(N, length(vals))
    λ = vals[i]
    global eigenvalue_log *= "  λ_$i = $(round(real(λ), digits=6)) + $(round(imag(λ), digits=6))i  (|λ|=$(round(abs(λ), digits=6)))\n"
end

global eigenvalue_log *= """

Expected (2D Ising universality):
  λ_σ ≈ 3.668, λ_ε ≈ 2.0, λ_T ≈ 1.0
"""

log_info(eigenvalue_log, paths["log"])
if verbosity > 0
    println(eigenvalue_log)
end

# Save results
result_data = Dict(
    "eigenvalues" => vals,
    "eigenvectors" => vecs,
    "config" => config,
    "A_crit" => A_crit_JU,
)
serialize(paths["data"], result_data)

println("\nResults saved to: $(paths["data"])")
println("Log saved to: $(paths["log"])")
