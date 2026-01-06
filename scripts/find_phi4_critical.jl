#!/usr/bin/env julia
# MODULE: find_phi4_critical.jl
# USAGE: julia --project=ekrgilttrnr ekrgilttrnr/scripts/find_phi4_critical.jl --mu_min 2.5 --mu_max 2.75
# DESCRIPTION: Automates the bisection search for the critical mass squared parameter in the Phi4 model using Gilt-TNR.
# INPUTS: --mu_min, --mu_max, --steps, --chi
# OUTPUTS: Prints the RG flow gap at each step and estimates the critical point.

using PyCall
using Printf
using ArgParse

# Add source directory to path
pushfirst!(PyVector(pyimport("sys")."path"), joinpath(@__DIR__, "../src/GiltTNR"))

# Import Python modules
const phi4_tools = pyimport("GiltTNR2D_Phi4")
const gilt_alg = pyimport("GiltTNR2D")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--mu_min"
            help = "Minimum mu^2"
            arg_type = Float64
            default = 2.5
        "--mu_max"
            help = "Maximum mu^2"
            arg_type = Float64
            default = 2.75
        "--steps"
            help = "Number of bisection steps"
            arg_type = Int
            default = 5
        "--chi"
            help = "Bond dimension"
            arg_type = Int
            default = 16
    end
    return parse_args(s)
end

function get_gap(mu_sq, chi)
    # Parameters
    phi4_pars = Dict(
        "mu_sq" => mu_sq,
        "lam" => 1.0,
        "kappa" => 1.0,
        "K" => 32,
        "D" => 16,
        "symmetry_tensors" => true
    )
    
    gilt_pars = Dict(
        "gilt_eps" => 6e-6,
        "cg_chis" => collect(1:chi),
        "cg_eps" => 1e-10,
        "verbosity" => 0,
        "rotate" => false
    )

    # Build tensor
    A = phi4_tools.get_initial_tensor_phi4(phi4_pars)
    tensor = A
    
    # Run RG for a few steps to see flow
    getitem = pyimport("operator").getitem
    
    final_gap = 0.0
    for step in 1:10
        result = gilt_alg.gilttnr_step(tensor, 0.0, gilt_pars)
        tensor = getitem(result, 0)
        
        # Measure Spectrum
        spectrum = phi4_tools.get_A_spectrum_phi4(tensor)
        gap = length(spectrum) > 1 ? spectrum[2] : 0.0
        final_gap = gap
        
        # Early exit if converged
        if gap < 1e-4 || abs(gap - 1.0) < 1e-4
            break
        end
    end
    
    return final_gap
end

function main()
    args = parse_commandline()
    
    mu_min = args["mu_min"]
    mu_max = args["mu_max"]
    steps = args["steps"]
    chi = args["chi"]
    
    println("Finding Critical Point for Phi4:")
    println("  Initial Range: [$mu_min, $mu_max]")
    println("  Steps: $steps")
    println("  Chi: $chi")
    println("-"^60)
    
    current_min = mu_min
    current_max = mu_max
    
    for i in 1:steps
        mid = (current_min + current_max) / 2
        gap = get_gap(mid, chi)
        
        phase = gap > 0.5 ? "Broken (Gap -> 1)" : "Symmetric (Gap -> 0)"
        @printf("Step %d: mu^2 = %.6f => Gap = %.6f [%s]\n", i, mid, gap, phase)
        
        if gap > 0.5
            # Broken phase (need larger mu^2 to get to Symmetric)
            # Wait, previous scan showed:
            # mu=2.5 -> Broken
            # mu=2.75 -> Symmetric
            # So Broken is LOWER mu^2. Symmetric is HIGHER mu^2.
            # If we are Broken, we are too low. We need to go higher.
            current_min = mid
        else
            # Symmetric phase (need smaller mu^2)
            current_max = mid
        end
    end
    
    println("-"^60)
    @printf("Critical Point Estimate: mu^2_c ≈ %.6f\n", (current_min + current_max) / 2)
end

main()
