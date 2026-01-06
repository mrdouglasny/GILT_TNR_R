# MODULE: phi4_exponents_plot.jl
# USAGE: julia --project=ekrgilttrnr ekrgilttrnr/scripts/phi4_exponents_plot.jl
# INPUTS: None (generates data from RG trajectory)
# OUTPUTS: ekrgilttrnr/data/phi4_scaling_dimensions.{png,pdf}
# DESCRIPTION: Generate RG trajectory and plot scaling dimensions for φ⁴ at criticality
#
# BASED ON: phi4-demo scripts, EKR Gilt-TNR implementation
# NEW: Standalone plotting script with corrected μ² = +2.731815

using PyCall
using Printf
using CairoMakie
using Serialization

# Load tools  
include("../src/Tools.jl")
include("../src/Phi4Tools.jl")

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

println("Generating φ⁴ RG trajectory at μ² = $mu_sq")
println("Parameters: χ = $chi, gilt_eps = $gilt_eps")
println("-"^50)

A = initial_tensor_phi4(phi4_pars)

steps = Int[]
x_sigma = Float64[]
x_epsilon = Float64[]

for step in 1:max_steps
    global A
    A, _ = py"gilttnr_step"(A, 0.0, gilt_pars)
    
    scaldims = py"get_scaldims_phi4"(A)
    
    if length(scaldims) >= 3
        push!(steps, step)
        push!(x_sigma, scaldims[2])
        push!(x_epsilon, scaldims[3])
        @printf("Step %2d: x_σ = %.4f, x_ε = %.4f\n", step, scaldims[2], scaldims[3])
    else
        println("Step $step: tensor collapsed")
        break
    end
end

########################################
# Create plot
########################################

println("\nGenerating plot...")

fig = Figure(size = (700, 500))

ax = Axis(fig[1, 1],
    xlabel = "RG Step",
    ylabel = "Scaling Dimension x",
    title = "φ⁴ Model: Approach to Ising CFT Fixed Point\n(μ² = $mu_sq, χ = $chi)",
    xticks = 1:length(steps),
    yticks = 0:0.25:2.0,
)

# Plot x_sigma (spin operator)
lines!(ax, steps, x_sigma, color = :blue, linewidth = 2, label = "x_σ (spin)")
scatter!(ax, steps, x_sigma, color = :blue, markersize = 10)

# Plot x_epsilon (energy operator)
lines!(ax, steps, x_epsilon, color = :red, linewidth = 2, label = "x_ε (energy)")
scatter!(ax, steps, x_epsilon, color = :red, markersize = 10)

# Add horizontal lines for exact Ising CFT values
hlines!(ax, [0.125], color = :blue, linestyle = :dash, linewidth = 1.5, label = "Ising x_σ = 0.125")
hlines!(ax, [1.0], color = :red, linestyle = :dash, linewidth = 1.5, label = "Ising x_ε = 1.0")

# Legend
axislegend(ax, position = :lt)

# Y limits
ylims!(ax, 0, 2.0)

# Save
mkpath("../data")
save("../data/phi4_scaling_dimensions.png", fig, px_per_unit = 2)
save("../data/phi4_scaling_dimensions.pdf", fig)

println("Plot saved to ekrgilttrnr/data/phi4_scaling_dimensions.{png,pdf}")

########################################
# Summary
########################################

# Find step with best x_epsilon match
best_step = argmin(abs.(x_epsilon .- 1.0))
println("\n" * "="^50)
println("SUMMARY: Best agreement at step $(steps[best_step])")
println("  x_σ = $(round(x_sigma[best_step], digits=4)) (Ising: 0.125, error: $(round(abs(x_sigma[best_step]-0.125), digits=4)))")
println("  x_ε = $(round(x_epsilon[best_step], digits=4)) (Ising: 1.0, error: $(round(abs(x_epsilon[best_step]-1.0), digits=4)))")
println("="^50)
