#!/usr/bin/env julia
# MODULE: phi4_exponents.jl
# USAGE: julia --project=ekrgilttrnr ekrgilttrnr/scripts/phi4_exponents.jl --mu_sq 2.731815 --chi 32 --steps 50
# DESCRIPTION: Runs Gilt-TNR for Phi4 model at critical point and extracts scaling dimensions.

using PyCall
using Printf
using ArgParse
using Serialization

# Add source directory to path
pushfirst!(PyVector(pyimport("sys")."path"), joinpath(@__DIR__, "../src/GiltTNR"))

# Import Python modules
const phi4_tools = pyimport("GiltTNR2D_Phi4")
const gilt_alg = pyimport("GiltTNR2D")
const operator = pyimport("operator")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--mu_sq"
            help = "Mass squared parameter"
            arg_type = Float64
            required = true
        "--chi"
            help = "Bond dimension"
            arg_type = Int
            default = 32
        "--steps"
            help = "Number of RG steps"
            arg_type = Int
            default = 50
        "--output"
            help = "Output file for results"
            arg_type = String
            default = "phi4_exponents.dat"
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    
    mu_sq = args["mu_sq"]
    chi = args["chi"]
    max_steps = args["steps"]
    output_file = args["output"]
    
    println("Running Phi4 Exponents Extraction:")
    println("  mu^2: $mu_sq")
    println("  Chi: $chi")
    println("  Steps: $max_steps")
    println("-"^60)
    
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
        "gilt_eps" => 1e-7,
        "cg_chis" => collect(1:chi),
        "cg_eps" => 1e-10,
        "verbosity" => 0,
        "rotate" => false
    )

    # Build tensor
    A = phi4_tools.get_initial_tensor_phi4(phi4_pars)
    tensor = A
    
    # Open output file
    open(output_file, "w") do io
        println(io, "# Step  x_sigma  x_epsilon  x_3  x_4 ...")
        
        for step in 1:max_steps
            # RG Step
            result = gilt_alg.gilttnr_step(tensor, 0.0, gilt_pars)
            tensor = operator.getitem(result, 0)
            
            # Measure Scaling Dimensions
            scaldims = phi4_tools.get_scaldims_phi4(tensor)
            
            # Print to console
            # Index 1 is Identity (x=0)
            # Index 2 is likely Sigma (x=0.125)
            # Index 3 is likely Epsilon (x=1.0) or subleading Sigma
            if length(scaldims) >= 3
                @printf("Step %d: x_1=%.6f  x_2=%.6f\n", step, scaldims[2], scaldims[3])
            else
                @printf("Step %d: x_1=%.6f\n", step, scaldims[2])
            end
            
            # Save to file
            print(io, "$step")
            for x in scaldims
                print(io, " $x")
            end
            println(io, "")
            flush(io)
        end
    end
    
    println("-"^60)
    println("Results saved to $output_file")
end

main()
