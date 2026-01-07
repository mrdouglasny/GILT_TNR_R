#!/usr/bin/env julia
# MODULE: potts3_exponents.jl
# USAGE: julia +1.10.4 --project=ekrgilttrnr ekrgilttrnr/scripts/potts3_exponents.jl [options]
# INPUTS: None (constructs tensor from parameters)
# OUTPUTS:
#   - stdout: scaling dimensions at each RG step
#   - ekrgilttrnr/data/potts3_exponents/<config>.csv
# DESCRIPTION: Extract scaling dimensions for 3-state Potts model using Gilt-TNR.
#
# BASED ON: ekrgilttrnr/scripts/phi4_exponents.jl, GiltTNR2D_Potts.py
# NEW IN THIS SCRIPT: Applies Gilt-TNR to 3-state Potts model for critical exponent extraction.
#
# 3-STATE POTTS CFT (c=4/5) EXPECTED VALUES:
#   x_identity = 0
#   x_spin (σ) = 2/15 ≈ 0.1333
#   x_energy (ε) = 4/5 = 0.8
#   x_spin2 = 4/3 ≈ 1.333
#
# RUNTIME: ~5-10 min for χ=16, ~30 min for χ=24

using PyCall
using Printf
using ArgParse
using Dates

# Setup Python path
pushfirst!(PyVector(pyimport("sys")."path"), joinpath(@__DIR__, "../GiltTNR"))

# Import Python modules
const potts_tools = pyimport("GiltTNR2D_Potts")
const gilt_alg = pyimport("GiltTNR2D")
const operator = pyimport("operator")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--q"
            help = "Number of Potts states"
            arg_type = Int
            default = 3
        "--relT"
            help = "Relative temperature (1.0 = critical)"
            arg_type = Float64
            default = 1.0
        "--chi"
            help = "Bond dimension"
            arg_type = Int
            default = 16
        "--steps"
            help = "Number of RG steps"
            arg_type = Int
            default = 30
        "--gilt_eps"
            help = "Gilt truncation threshold"
            arg_type = Float64
            default = 1e-6
        "--output"
            help = "Output CSV file"
            arg_type = String
            default = ""
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()

    q = args["q"]
    relT = args["relT"]
    chi = args["chi"]
    max_steps = args["steps"]
    gilt_eps = args["gilt_eps"]

    # Default output file
    output_file = args["output"]
    if output_file == ""
        mkpath("ekrgilttrnr/data/potts3_exponents")
        output_file = "ekrgilttrnr/data/potts3_exponents/q$(q)_chi$(chi)_relT$(relT).csv"
    end

    # Critical point
    beta_c = log(1 + sqrt(q))

    println("="^70)
    println("3-State Potts Model: Scaling Dimension Extraction via Gilt-TNR")
    println("="^70)
    println()
    println("Parameters:")
    println("  q (states):     $q")
    println("  relT:           $relT (1.0 = critical)")
    println("  β_c (exact):    $(round(beta_c, digits=6))")
    println("  β (actual):     $(round(beta_c/relT, digits=6))")
    println("  χ (bond dim):   $chi")
    println("  Gilt ε:         $gilt_eps")
    println("  RG steps:       $max_steps")
    println()
    println("Expected CFT scaling dimensions (c=4/5):")
    println("  x_identity = 0")
    println("  x_spin (σ) = 2/15 ≈ 0.1333")
    println("  x_energy (ε) = 4/5 = 0.8")
    println()
    println("-"^70)

    # Build initial tensor
    potts_pars = Dict(
        "q" => q,
        "relT" => relT,
        "symmetry_tensors" => false
    )

    gilt_pars = Dict(
        "gilt_eps" => gilt_eps,
        "cg_chis" => collect(1:chi),
        "cg_eps" => 1e-10,
        "verbosity" => 0,
        "rotate" => false
    )

    println("Building initial tensor...")
    A = potts_tools.get_initial_tensor_potts_relT(potts_pars)
    tensor = A
    log_fact = 0.0

    # Storage for results
    results = []

    println()
    println(@sprintf("%-6s  %-10s  %-10s  %-10s  %-10s  %-10s",
                     "Step", "x_1", "x_2", "x_3", "x_4", "x_5"))
    println("-"^70)

    for step in 1:max_steps
        # RG Step
        result = gilt_alg.gilttnr_step(tensor, log_fact, gilt_pars)
        tensor = operator.getitem(result, 0)
        log_fact = operator.getitem(result, 1)

        # Extract scaling dimensions
        scaldims = potts_tools.get_scaldims_potts(tensor)
        scaldims = convert(Vector{Float64}, scaldims)

        # Pad to 5 elements
        while length(scaldims) < 6
            push!(scaldims, NaN)
        end

        # x_0 = 0 (identity), so x_1 is first non-trivial
        x_vals = scaldims[2:6]

        push!(results, (step=step, x_vals=x_vals))

        @printf("%-6d  %-10.6f  %-10.6f  %-10.6f  %-10.6f  %-10.6f\n",
                step, x_vals...)

        # Check convergence
        if step > 10
            # Compare with CFT predictions
            x_spin_expected = 2/15
            x_energy_expected = 4/5

            x_spin_error = abs(x_vals[1] - x_spin_expected)
            x_energy_error = abs(x_vals[2] - x_energy_expected)

            if x_spin_error < 0.01 && x_energy_error < 0.05
                # Good convergence
            end
        end
    end

    println("-"^70)
    println()

    # Final analysis
    if length(results) > 5
        # Average over last few steps
        last_n = min(5, length(results))
        x1_avg = mean([r.x_vals[1] for r in results[end-last_n+1:end]])
        x2_avg = mean([r.x_vals[2] for r in results[end-last_n+1:end]])

        println("Final scaling dimensions (averaged over last $last_n steps):")
        println(@sprintf("  x_1 = %.6f  (expected σ: %.6f, error: %.2f%%)",
                        x1_avg, 2/15, 100*abs(x1_avg - 2/15)/(2/15)))
        println(@sprintf("  x_2 = %.6f  (expected ε: %.6f, error: %.2f%%)",
                        x2_avg, 4/5, 100*abs(x2_avg - 4/5)/(4/5)))
    end

    # Save results to simple text file
    println()
    println("Saving results to $output_file")
    open(output_file, "w") do io
        println(io, "# step x_1 x_2 x_3 x_4 x_5")
        for r in results
            println(io, "$(r.step) $(join(r.x_vals, " "))")
        end
    end

    println()
    println("="^70)
    println("COMPLETED")
    println("="^70)
end

# Helper
function mean(arr)
    sum(arr) / length(arr)
end

main()
