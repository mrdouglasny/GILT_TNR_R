# MODULE: scan_phi4_flow.jl
# USAGE: julia --project=ekrgilttrnr ekrgilttrnr/scripts/scan_phi4_flow.jl --mu_center -3.8e-7 --range 1e-7 --steps 5
# DESCRIPTION: Scans RG flow for Phi4 model around a central mu^2 to visualize phase splitting.

using ArgParse
using Serialization
using Printf
using LinearAlgebra

# Load ProjectUtils
include("../../src/julia/ProjectUtils.jl")
using .ProjectUtils

# Load tools
include("../src/Tools.jl")
include("../src/Phi4Tools.jl")
include("../src/GaugeFixing.jl")
include("../src/Phi4GaugeFixing.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--mu_center"
        help = "Center mu^2 for scan"
        arg_type = Float64
        required = true
        "--range"
        help = "Range of scan (+/-)"
        arg_type = Float64
        required = true
        "--points"
        help = "Number of points in scan"
        arg_type = Int
        default = 5
        "--chi"
        help = "Bond dimension"
        arg_type = Int
        default = 16
        "--max_rg_steps"
        help = "Max RG steps to track"
        arg_type = Int
        default = 20
        "--lam"
        default = 1.0
        arg_type = Float64
        "--kappa"
        default = 1.0
        arg_type = Float64
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    
    mu_center = args["mu_center"]
    r = args["range"]
    points = args["points"]
    chi = args["chi"]
    
    # Generate mu values
    mus = range(mu_center - r, mu_center + r, length=points)
    
    println("Scanning RG flow for Phi4:")
    println("  Center mu^2: $mu_center")
    println("  Range: +/- $r")
    println("  Points: $points")
    println("  Chi: $chi")
    println("-"^60)
    println("Step | " * join([@sprintf("mu=%+.2e", m) for m in mus], " | "))
    println("-"^60)

    # Initialize tensors
    tensors = []
    gilt_pars = Dict(
        "gilt_eps" => 6e-6,
        "cg_chis" => collect(1:chi),
        "cg_eps" => 1e-10,
        "verbosity" => 0,
        "rotate" => false
    )

    for mu in mus
        phi4_pars = Dict(
            "mu_sq" => mu,
            "lam" => args["lam"],
            "kappa" => args["kappa"],
            "K" => 32,
            "D" => 16,
            "symmetry_tensors" => true
        )
        push!(tensors, to_python_tensor(initial_tensor_phi4(phi4_pars)))
    end

    # Run RG
    getitem = pyimport("operator").getitem
    
    for step in 1:args["max_rg_steps"]
        gaps = []
        for i in 1:length(tensors)
            # RG Step
            result = pycall(py"gilttnr_step", PyObject, tensors[i], 0.0, gilt_pars)
            tensors[i] = pycall(getitem, PyObject, result, 0)
            
            # Measure Spectrum
            spectrum = py"get_A_spectrum_phi4"(tensors[i])
            gap = length(spectrum) > 1 ? spectrum[2] : 0.0
            push!(gaps, gap)
        end
        
        # Print row
        row_str = @sprintf("%4d | ", step)
        row_str *= join([@sprintf("%9.6f", g) for g in gaps], " | ")
        println(row_str)
        
        # Check if all converged
        if all(g -> (g < 1e-4 || abs(g - 1.0) < 1e-4), gaps)
            println("-"^60)
            println("All trajectories converged to trivial fixed points.")
            break
        end
    end
end

main()
