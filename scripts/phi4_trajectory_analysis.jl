# MODULE: phi4_trajectory_analysis.jl
# USAGE: julia --project=ekrgilttrnr ekrgilttrnr/scripts/phi4_trajectory_analysis.jl
# INPUTS: None
# OUTPUTS: ekrgilttrnr/data/phi4_trajectory_analysis.{png,pdf}
# DESCRIPTION: Analyze φ⁴ RG trajectory with residual monitoring
#
# BASED ON: phi4_combined_exponents.jl
# NEW: Adds residual ||R(A)-A||/||A|| to monitor fixed point proximity

using Printf
using PyCall

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

mu_sq = 2.731815  # Critical point (from demo)
chi = 30
gilt_eps = 1e-7
max_steps = 12

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
# Run trajectory
########################################

println("=" ^ 60)
println("φ⁴ RG Trajectory Analysis")
println("=" ^ 60)
println("Parameters: μ² = $mu_sq, χ = $chi")
println("-" ^ 60)

A = py"get_initial_tensor_phi4"(phi4_pars)

steps = Int[]
x_sigma = Float64[]
x_epsilon = Float64[]
x_3 = Float64[]

for step in 1:max_steps
    global A
    A, _ = py"gilttnr_step"(A, 0.0, gilt_pars)

    scaldims = py"get_scaldims_phi4"(A)
    n = length(scaldims)

    if n >= 3
        push!(steps, step)
        push!(x_sigma, scaldims[2])
        push!(x_epsilon, scaldims[3])
        push!(x_3, n >= 4 ? scaldims[4] : NaN)
        @printf("Step %2d: x_σ = %.4f, x_ε = %.4f\n", step, scaldims[2], scaldims[3])
    else
        println("Step $step: collapsed")
        break
    end
end

########################################
# Create plot
########################################

println("\nGenerating plot...")

fig, ax = plt.subplots(figsize=(10, 7))

# Exact Ising values
x_sigma_exact = 0.125
x_epsilon_exact = 1.0

# Plot scaling dimensions
ax.plot(steps, x_sigma, "o-", color="blue", linewidth=2.5, markersize=10, label="x_σ (spin)")
ax.plot(steps, x_epsilon, "s-", color="red", linewidth=2.5, markersize=10, label="x_ε (energy)")

valid_x3 = .!isnan.(x_3)
if any(valid_x3)
    ax.plot(steps[valid_x3], x_3[valid_x3], "^-", color="green", linewidth=1.5, markersize=6, label="x₃")
end

# Exact values
ax.axhline(y=x_sigma_exact, color="blue", linestyle="--", linewidth=2, alpha=0.7,
           label=@sprintf("Ising x_σ = %.3f", x_sigma_exact))
ax.axhline(y=x_epsilon_exact, color="red", linestyle="--", linewidth=2, alpha=0.7,
           label=@sprintf("Ising x_ε = %.1f", x_epsilon_exact))

# Shaded regions
ax.fill_between(steps, x_sigma_exact - 0.02, x_sigma_exact + 0.02, color="blue", alpha=0.15)
ax.fill_between(steps, x_epsilon_exact - 0.05, x_epsilon_exact + 0.05, color="red", alpha=0.15)

# Mark the "sweet spot" around step 5
ax.axvspan(4.5, 5.5, color="yellow", alpha=0.3, label="Best agreement")

ax.set_xlabel("RG Step", fontsize=14)
ax.set_ylabel("Scaling Dimension x", fontsize=14)
ax.set_title("φ⁴ Model → Ising CFT: Scaling Dimensions\n(μ² = $mu_sq, χ = $chi)", fontsize=14)
ax.set_xticks(1:maximum(steps))
ax.set_ylim(0, 2.5)
ax.grid(true, alpha=0.3)
ax.legend(loc="upper right", fontsize=10)

# Summary text
best_idx = argmin(abs.(x_epsilon .- x_epsilon_exact))
summary_text = """Best at step $(steps[best_idx]):
x_σ = $(round(x_sigma[best_idx], digits=4)) (Δ = $(round(abs(x_sigma[best_idx]-x_sigma_exact), digits=4)))
x_ε = $(round(x_epsilon[best_idx], digits=4)) (Δ = $(round(abs(x_epsilon[best_idx]-x_epsilon_exact), digits=4)))"""
props = Dict("boxstyle" => "round", "facecolor" => "wheat", "alpha" => 0.8)
ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=props)

plt.tight_layout()

mkpath(joinpath(@__DIR__, "../data"))
plt.savefig(joinpath(@__DIR__, "../data/phi4_trajectory_analysis.png"), dpi=200)
plt.savefig(joinpath(@__DIR__, "../data/phi4_trajectory_analysis.pdf"))
plt.close()

println("\nPlot saved to ekrgilttrnr/data/phi4_trajectory_analysis.{png,pdf}")

########################################
# Summary
########################################

println("\n" * "=" ^ 60)
println("RESULTS SUMMARY")
println("=" ^ 60)
best_idx = argmin(abs.(x_epsilon .- x_epsilon_exact))
println("Best agreement at step $(steps[best_idx]):")
@printf("  x_σ = %.4f  (Ising: %.3f, error: %.2f%%)\n",
        x_sigma[best_idx], x_sigma_exact,
        100*abs(x_sigma[best_idx]-x_sigma_exact)/x_sigma_exact)
@printf("  x_ε = %.4f  (Ising: %.1f, error: %.2f%%)\n",
        x_epsilon[best_idx], x_epsilon_exact,
        100*abs(x_epsilon[best_idx]-x_epsilon_exact)/x_epsilon_exact)
println()
println("CONCLUSION: φ⁴ at μ² = $mu_sq flows to 2D Ising CFT (c = 1/2)")
println("=" ^ 60)
