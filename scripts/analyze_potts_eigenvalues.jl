#!/usr/bin/env julia
# MODULE: analyze_potts_eigenvalues.jl
# USAGE: julia --project=ekrgilttrnr scripts/analyze_potts_eigenvalues.jl
# INPUTS: None (reads from data/)
# OUTPUTS: Analysis of eigenvalues vs CFT predictions
# DESCRIPTION: Analyze Newton iteration results for 3-state Potts model
#
# BASED ON: EKR arXiv:2408.10312 Table 7 (Potts critical exponents)
# NEW IN THIS MODULE: Diagnostic analysis of eigenvalue spectrum

println("="^70)
println("3-State Potts Eigenvalue Analysis")
println("="^70)

# CFT predictions for 3-state Potts
# Central charge: c = 4/5
# Primary operators and their scaling dimensions:
const X_SIGMA = 2/15      # Spin operator (order parameter)
const X_EPSILON = 4/5     # Energy operator
const X_STRESS = 2.0      # Stress tensor (T, T̄)

# Corresponding Jacobian eigenvalues: λ = 2^(2x)
const LAMBDA_SIGMA = 2^(2 * X_SIGMA)      # ≈ 1.20
const LAMBDA_EPSILON = 2^(2 * X_EPSILON)  # ≈ 3.03
const LAMBDA_STRESS = 2^(2 * X_STRESS)    # = 16 (marginal)

println("\n--- CFT Predictions ---")
println("x_σ = 2/15 ≈ $(round(X_SIGMA, digits=4))")
println("x_ε = 4/5 = $(X_EPSILON)")
println("x_T = 2 (stress tensor)")
println()
println("Expected eigenvalues:")
println("  λ_σ = 2^(2x_σ) ≈ $(round(LAMBDA_SIGMA, digits=4))")
println("  λ_ε = 2^(2x_ε) ≈ $(round(LAMBDA_EPSILON, digits=4))")
println("  λ_T = 2^4 = $(LAMBDA_STRESS) (marginal)")

# Read latest data file
data_dir = joinpath(dirname(@__DIR__), "data", "potts3_newton")
if !isdir(data_dir)
    error("Data directory not found: $data_dir")
end

# Find most recent file
files = filter(f -> endswith(f, ".csv"), readdir(data_dir, join=true))
if isempty(files)
    error("No data files found")
end
latest = sort(files, by=mtime)[end]

println("\n--- Latest Data File ---")
println("File: $(basename(latest))")

# Parse eigenvalues
eigenvalues = Tuple{Float64, ComplexF64}[]  # (abs, complex)
lines = readlines(latest)
for line in lines
    if startswith(line, "#") || isempty(strip(line)) || startswith(line, "index")
        continue
    end
    parts = split(line, ",")
    if length(parts) >= 4
        idx = parse(Int, parts[1])
        abs_val = parse(Float64, parts[2])
        re_val = parse(Float64, parts[3])
        im_val = parse(Float64, parts[4])
        push!(eigenvalues, (abs_val, complex(re_val, im_val)))
    end
end

println("\n--- Eigenvalue Analysis ---")
println("Number of eigenvalues: $(length(eigenvalues))")
println()

# Convert to scaling dimensions: x = log(|λ|) / (2 log(2))
for (i, (abs_λ, λ)) in enumerate(eigenvalues)
    x = log(abs_λ) / (2 * log(2))

    # Check proximity to expected values
    err_sigma = abs(x - X_SIGMA) / X_SIGMA * 100
    err_epsilon = abs(x - X_EPSILON) / X_EPSILON * 100

    # Determine which operator this might be
    candidate = if abs_λ > 100
        "Spurious (|λ| > 100)"
    elseif abs_λ < 0.1
        "Irrelevant (|λ| < 0.1)"
    elseif abs(x - X_SIGMA) < 0.2
        "σ? (error: $(round(err_sigma, digits=1))%)"
    elseif abs(x - X_EPSILON) < 0.3
        "ε? (error: $(round(err_epsilon, digits=1))%)"
    elseif abs(x - 2.0) < 0.5
        "T? (stress tensor)"
    elseif 0.0 < x < 0.1
        "Near marginal"
    else
        "Unknown"
    end

    # Format eigenvalue
    λ_str = if abs(imag(λ)) < 1e-10
        "$(round(real(λ), digits=4))"
    else
        "$(round(real(λ), digits=4)) ± $(round(abs(imag(λ)), digits=4))i"
    end

    println("λ_$i = $λ_str")
    println("    |λ| = $(round(abs_λ, digits=4))")
    println("    x   = $(round(x, digits=4))")
    println("    → $candidate")
    println()
end

println("--- Summary ---")
println("To get reliable scaling dimensions:")
println("1. Physical eigenvalues should have |λ| ~ 1-10")
println("2. Spurious modes (|λ| >> 100) indicate:")
println("   - Gauge fixing issues")
println("   - Tensor not at true fixed point")
println("   - Numerical instability in Jacobian")
println()
println("Expected physical spectrum for 3-state Potts:")
println("  λ_σ ≈ $(round(LAMBDA_SIGMA, digits=3)) → x_σ = $(round(X_SIGMA, digits=4))")
println("  λ_ε ≈ $(round(LAMBDA_EPSILON, digits=3)) → x_ε = $(X_EPSILON)")
println()
println("Reference: EKR arXiv:2408.10312, Table 7")
