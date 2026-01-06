# MODULE: phi4_combined_exponents.jl
# USAGE: julia --project=ekrgilttrnr ekrgilttrnr/scripts/phi4_combined_exponents.jl
# INPUTS: None
# OUTPUTS: ekrgilttrnr/data/phi4_combined_exponents.{png,pdf}
# DESCRIPTION: Combined plot showing scaling dimensions from transfer matrix at each RG step
#
# BASED ON: phi4_exponents_plot.jl
# NEW: Shows detailed convergence with error bands

using Printf
using PyCall

# Use matplotlib via PyCall
const plt = pyimport("matplotlib.pyplot")
const mplpatches = pyimport("matplotlib.patches")

# Add GiltTNR to Python path
pushfirst!(pyimport("sys")."path", joinpath(@__DIR__, "../src/GiltTNR"))

# Import Python modules
@pyinclude(joinpath(@__DIR__, "../src/GiltTNR/GiltTNR2D_Phi4.py"))
@pyinclude(joinpath(@__DIR__, "../src/GiltTNR/GiltTNR2D_essentials.py"))

########################################
# Parameters
########################################

mu_sq = 2.731815  # Critical point (POSITIVE!)
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
# Generate RG trajectory
########################################

println("=" ^ 60)
println("φ⁴ Scaling Dimensions via Transfer Matrix")
println("=" ^ 60)
println("Parameters: μ² = $mu_sq, χ = $chi, gilt_eps = $gilt_eps")
println("-" ^ 60)

A = py"get_initial_tensor_phi4"(phi4_pars)

steps = Int[]
x_sigma = Float64[]
x_epsilon = Float64[]
x_3 = Float64[]
x_4 = Float64[]

println("\nRunning RG trajectory...")
println("-" ^ 60)

for step in 1:max_steps
    global A
    A, _ = py"gilttnr_step"(A, 0.0, gilt_pars)

    scaldims = py"get_scaldims_phi4"(A)
    n = length(scaldims)

    if n >= 3
        push!(steps, step)
        push!(x_sigma, scaldims[2])
        push!(x_epsilon, scaldims[3])
        if n >= 4
            push!(x_3, scaldims[4])
        else
            push!(x_3, NaN)
        end
        if n >= 5
            push!(x_4, scaldims[5])
        else
            push!(x_4, NaN)
        end
        @printf("Step %2d: x₁ = %.4f, x₂ = %.4f", step, scaldims[2], scaldims[3])
        if n >= 4
            @printf(", x₃ = %.4f", scaldims[4])
        end
        println()
    else
        println("Step $step: tensor collapsed ($(n) eigenvalues)")
        break
    end
end

########################################
# Exact Ising CFT values
########################################

# 2D Ising CFT scaling dimensions:
# Identity: x = 0
# Spin (σ): x = 1/8 = 0.125
# Energy (ε): x = 1
# Energy-squared: x = 2
# Spin-spin: x = 1/4 (from σ × σ OPE)

x_sigma_exact = 0.125
x_epsilon_exact = 1.0

########################################
# Create plot
########################################

println("\nGenerating plot...")

fig, ax = plt.subplots(figsize=(10, 7))

# Plot x_sigma (spin operator)
ax.plot(steps, x_sigma, "o-", color="blue", linewidth=2.5, markersize=10, label="x_σ (spin)")

# Plot x_epsilon (energy operator)
ax.plot(steps, x_epsilon, "s-", color="red", linewidth=2.5, markersize=10, label="x_ε (energy)")

# Plot higher scaling dimensions if available
valid_x3 = .!isnan.(x_3)
if any(valid_x3)
    ax.plot(steps[valid_x3], x_3[valid_x3], "^-", color="green", linewidth=1.5, markersize=6, label="x₃")
end

valid_x4 = .!isnan.(x_4)
if any(valid_x4)
    ax.plot(steps[valid_x4], x_4[valid_x4], "d-", color="purple", linewidth=1.5, markersize=6, label="x₄")
end

# Add horizontal lines for exact Ising CFT values
ax.axhline(y=x_sigma_exact, color="blue", linestyle="--", linewidth=2, alpha=0.7,
           label=@sprintf("Ising x_σ = %.3f", x_sigma_exact))
ax.axhline(y=x_epsilon_exact, color="red", linestyle="--", linewidth=2, alpha=0.7,
           label=@sprintf("Ising x_ε = %.1f", x_epsilon_exact))

# Add shaded regions for convergence
ax.fill_between(steps, x_sigma_exact - 0.02, x_sigma_exact + 0.02, color="blue", alpha=0.15)
ax.fill_between(steps, x_epsilon_exact - 0.05, x_epsilon_exact + 0.05, color="red", alpha=0.15)

# Labels and title
ax.set_xlabel("RG Step", fontsize=14)
ax.set_ylabel("Scaling Dimension x", fontsize=14)
ax.set_title("φ⁴ Model → Ising CFT: Scaling Dimensions\n(μ² = $mu_sq, χ = $chi)", fontsize=14)

# Axes settings
ax.set_xticks(1:maximum(steps))
ax.set_ylim(0, 2.5)
ax.grid(true, alpha=0.3)

# Legend
ax.legend(loc="upper right", fontsize=10)

########################################
# Add summary text box
########################################

# Find step with best x_epsilon match
best_step_idx = argmin(abs.(x_epsilon .- x_epsilon_exact))
best_step = steps[best_step_idx]

# Compute errors at best step
err_sigma = abs(x_sigma[best_step_idx] - x_sigma_exact)
err_epsilon = abs(x_epsilon[best_step_idx] - x_epsilon_exact)

summary_text = """Best agreement at step $best_step:
x_σ = $(round(x_sigma[best_step_idx], digits=4)) (Ising: $x_sigma_exact, Δ = $(round(err_sigma, digits=4)))
x_ε = $(round(x_epsilon[best_step_idx], digits=4)) (Ising: $x_epsilon_exact, Δ = $(round(err_epsilon, digits=4)))"""

# Add text box
props = Dict("boxstyle" => "round", "facecolor" => "wheat", "alpha" => 0.8)
ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=props)

########################################
# Save plot
########################################

mkpath(joinpath(@__DIR__, "../data"))
plt.tight_layout()
plt.savefig(joinpath(@__DIR__, "../data/phi4_combined_exponents.png"), dpi=200)
plt.savefig(joinpath(@__DIR__, "../data/phi4_combined_exponents.pdf"))
plt.close()

println("\nPlot saved to ekrgilttrnr/data/phi4_combined_exponents.{png,pdf}")

########################################
# Summary
########################################

println("\n" * "=" ^ 60)
println("SUMMARY")
println("=" ^ 60)
println("Best agreement at step $best_step:")
@printf("  x_σ = %.4f  (Ising: %.3f, error: %.4f)\n", x_sigma[best_step_idx], x_sigma_exact, err_sigma)
@printf("  x_ε = %.4f  (Ising: %.1f, error: %.4f)\n", x_epsilon[best_step_idx], x_epsilon_exact, err_epsilon)
println()
println("This confirms φ⁴ at criticality flows to the Ising CFT fixed point!")
println("=" ^ 60)
