# MODULE: phi4_newton_corrected.jl
# USAGE: julia --project=ekrgilttrnr ekrgilttrnr/scripts/phi4_newton_corrected.jl
# INPUTS: None
# OUTPUTS: ekrgilttrnr/data/phi4_newton_corrected.{png,pdf}
# DESCRIPTION: Newton-corrected RG trajectory - stays on critical manifold
#
# BASED ON: phi4_combined_exponents.jl, Newton method from EKR
# NEW: Applies Newton correction at each step to prevent drift from criticality

using Printf
using PyCall
using LinearAlgebra

# Use matplotlib via PyCall
const plt = pyimport("matplotlib.pyplot")
const np = pyimport("numpy")

# Add GiltTNR to Python path
pushfirst!(pyimport("sys")."path", joinpath(@__DIR__, "../src/GiltTNR"))

# Import Python modules
@pyinclude(joinpath(@__DIR__, "../src/GiltTNR/GiltTNR2D_Phi4.py"))
@pyinclude(joinpath(@__DIR__, "../src/GiltTNR/GiltTNR2D_essentials.py"))

const Tensor = pyimport("tensors").Tensor
const gilt_module = pyimport("GiltTNR2D_essentials")

########################################
# Parameters
########################################

mu_sq = 2.731815  # Starting critical point estimate
chi = 30
gilt_eps = 1e-7
max_steps = 12
newton_tol = 1e-8
newton_max_iter = 5

gilt_pars = Dict(
    "gilt_eps" => gilt_eps,
    "cg_chis" => collect(1:chi),
    "cg_eps" => 1e-10,
    "verbosity" => 0,
    "rotate" => false,
)

phi4_pars = Dict(
    "mu_sq" => mu_sq,
    "lam" => 1.0,
    "kappa" => 1.0,
    "K" => 32,
    "D" => 16,
    "symmetry_tensors" => true
)

########################################
# Helper functions
########################################

function tensor_to_array(A)
    if A isa Array
        return A
    else
        return convert(Array{Float64}, A.to_ndarray())
    end
end

function array_to_tensor(arr)
    arr_np = np.array(arr)
    return pycall(Tensor.from_ndarray, PyObject, arr_np)
end

function apply_rg(A)
    # If A is a Julia array, convert to Tensor first
    if A isa Array
        A = array_to_tensor(A)
    end
    result = pycall(gilt_module.gilttnr_step, PyObject, A, 0.0, gilt_pars)
    getitem = pyimport("operator").getitem
    return pycall(getitem, PyObject, result, 0)
end

function tensor_norm(A)
    arr = tensor_to_array(A)
    return sqrt(sum(arr.^2))
end

function ensure_tensor(A)
    if A isa Array
        return array_to_tensor(A)
    else
        return A
    end
end

function tensor_diff_norm(A, B)
    arr_A = tensor_to_array(A)
    arr_B = tensor_to_array(B)
    # Handle shape mismatch
    sA, sB = size(arr_A), size(arr_B)
    s_min = Tuple(min(sA[i], sB[i]) for i in 1:4)
    diff = arr_A[1:s_min[1], 1:s_min[2], 1:s_min[3], 1:s_min[4]] -
           arr_B[1:s_min[1], 1:s_min[2], 1:s_min[3], 1:s_min[4]]
    return sqrt(sum(diff.^2))
end

"""
Compute residual ||R(A) - A|| / ||A|| to monitor fixed point proximity.

At the true fixed point A*, we have R(A*) = A*, so residual = 0.
Away from fixed point, residual grows as we iterate.
"""
function compute_residual(A)
    A_rg = apply_rg(A)
    residual = tensor_diff_norm(A_rg, A)
    norm_A = tensor_norm(A)
    return residual / norm_A
end

########################################
# Run trajectories: with and without Newton correction
########################################

println("=" ^ 60)
println("φ⁴ Newton-Corrected RG Trajectory")
println("=" ^ 60)
println("Parameters: μ² = $mu_sq, χ = $chi")
println("-" ^ 60)

# Trajectory WITHOUT Newton correction (baseline)
println("\n--- Trajectory WITHOUT Newton correction ---")
A_plain = py"get_initial_tensor_phi4"(phi4_pars)

steps_plain = Int[]
x_sigma_plain = Float64[]
x_epsilon_plain = Float64[]

for step in 1:max_steps
    global A_plain
    A_plain, _ = py"gilttnr_step"(A_plain, 0.0, gilt_pars)

    scaldims = py"get_scaldims_phi4"(A_plain)
    if length(scaldims) >= 3
        push!(steps_plain, step)
        push!(x_sigma_plain, scaldims[2])
        push!(x_epsilon_plain, scaldims[3])
        @printf("Step %2d: x_σ = %.4f, x_ε = %.4f\n", step, scaldims[2], scaldims[3])
    else
        println("Step $step: tensor collapsed")
        break
    end
end

# Trajectory WITH Newton correction
println("\n--- Trajectory WITH Newton correction ---")
A_newton = py"get_initial_tensor_phi4"(phi4_pars)

steps_newton = Int[]
x_sigma_newton = Float64[]
x_epsilon_newton = Float64[]
residuals = Float64[]

for step in 1:max_steps
    global A_newton

    # Apply RG step
    A_newton, _ = py"gilttnr_step"(A_newton, 0.0, gilt_pars)

    # Apply Newton correction to stay near fixed point
    A_newton, n_iter, residual = newton_correct(A_newton; α=0.3, max_iter=3)

    scaldims = py"get_scaldims_phi4"(A_newton)
    if length(scaldims) >= 3
        push!(steps_newton, step)
        push!(x_sigma_newton, scaldims[2])
        push!(x_epsilon_newton, scaldims[3])
        push!(residuals, residual)
        @printf("Step %2d: x_σ = %.4f, x_ε = %.4f (Newton iter=%d, res=%.2e)\n",
                step, scaldims[2], scaldims[3], n_iter, residual)
    else
        println("Step $step: tensor collapsed")
        break
    end
end

########################################
# Create comparison plot
########################################

println("\nGenerating comparison plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Exact Ising values
x_sigma_exact = 0.125
x_epsilon_exact = 1.0

# Left panel: x_sigma comparison
ax1.plot(steps_plain, x_sigma_plain, "o--", color="blue", alpha=0.5,
         linewidth=1.5, markersize=8, label="Plain RG")
ax1.plot(steps_newton, x_sigma_newton, "o-", color="blue",
         linewidth=2.5, markersize=10, label="Newton-corrected")
ax1.axhline(y=x_sigma_exact, color="blue", linestyle=":", linewidth=2,
            label="Ising x_σ = 0.125")
ax1.fill_between([1, max_steps], x_sigma_exact - 0.02, x_sigma_exact + 0.02,
                 color="blue", alpha=0.15)
ax1.set_xlabel("RG Step", fontsize=14)
ax1.set_ylabel("x_σ (spin)", fontsize=14)
ax1.set_title("Spin Scaling Dimension", fontsize=14)
ax1.set_ylim(0, 0.5)
ax1.legend(fontsize=10)
ax1.grid(true, alpha=0.3)

# Right panel: x_epsilon comparison
ax2.plot(steps_plain, x_epsilon_plain, "s--", color="red", alpha=0.5,
         linewidth=1.5, markersize=8, label="Plain RG")
ax2.plot(steps_newton, x_epsilon_newton, "s-", color="red",
         linewidth=2.5, markersize=10, label="Newton-corrected")
ax2.axhline(y=x_epsilon_exact, color="red", linestyle=":", linewidth=2,
            label="Ising x_ε = 1.0")
ax2.fill_between([1, max_steps], x_epsilon_exact - 0.05, x_epsilon_exact + 0.05,
                 color="red", alpha=0.15)
ax2.set_xlabel("RG Step", fontsize=14)
ax2.set_ylabel("x_ε (energy)", fontsize=14)
ax2.set_title("Energy Scaling Dimension", fontsize=14)
ax2.set_ylim(0.5, 2.0)
ax2.legend(fontsize=10)
ax2.grid(true, alpha=0.3)

fig.suptitle("φ⁴ → Ising CFT: Effect of Newton Correction\n(μ² = $mu_sq, χ = $chi)",
             fontsize=14)

plt.tight_layout()

mkpath(joinpath(@__DIR__, "../data"))
plt.savefig(joinpath(@__DIR__, "../data/phi4_newton_corrected.png"), dpi=200)
plt.savefig(joinpath(@__DIR__, "../data/phi4_newton_corrected.pdf"))
plt.close()

println("\nPlot saved to ekrgilttrnr/data/phi4_newton_corrected.{png,pdf}")

########################################
# Summary
########################################

println("\n" * "=" ^ 60)
println("SUMMARY")
println("=" ^ 60)

# Find best step for each method
if !isempty(x_epsilon_plain)
    best_plain = argmin(abs.(x_epsilon_plain .- x_epsilon_exact))
    println("Plain RG best (step $(steps_plain[best_plain])):")
    @printf("  x_σ = %.4f (error: %.4f)\n", x_sigma_plain[best_plain],
            abs(x_sigma_plain[best_plain] - x_sigma_exact))
    @printf("  x_ε = %.4f (error: %.4f)\n", x_epsilon_plain[best_plain],
            abs(x_epsilon_plain[best_plain] - x_epsilon_exact))
end

if !isempty(x_epsilon_newton)
    best_newton = argmin(abs.(x_epsilon_newton .- x_epsilon_exact))
    println("\nNewton-corrected best (step $(steps_newton[best_newton])):")
    @printf("  x_σ = %.4f (error: %.4f)\n", x_sigma_newton[best_newton],
            abs(x_sigma_newton[best_newton] - x_sigma_exact))
    @printf("  x_ε = %.4f (error: %.4f)\n", x_epsilon_newton[best_newton],
            abs(x_epsilon_newton[best_newton] - x_epsilon_exact))

    # Check stability at later steps
    if length(x_epsilon_newton) >= 10
        println("\nStability check at step 10:")
        @printf("  x_σ = %.4f, x_ε = %.4f\n", x_sigma_newton[10], x_epsilon_newton[10])
    end
end

println("=" ^ 60)
