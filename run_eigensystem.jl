#!/usr/bin/env julia
# Wrapper script to run eigensystem.jl from the correct directory

# Change to src directory where Tools.jl can find GiltTNR
cd(joinpath(@__DIR__, "src"))

# Include the necessary modules
include("Tools.jl")
include("GaugeFixing.jl")
include("KrylovTechnical.jl")

# Change to scripts directory and run eigensystem
cd(joinpath(@__DIR__, "scripts"))

# Parse command line arguments
using ArgParse

settings = ArgParseSettings()
@add_arg_table! settings begin
	"--chi"
	help = "The bond dimension"
	arg_type = Int64
	default = 30
	"--gilt_eps"
	help = "The threshold used in the GILT algorithm"
	arg_type = Float64
	default = 6e-6
	"--N"
	help = "Number of eigenvalues to compute"
	arg_type = Int64
	default = 10
	"--krylovdim"
	help = "Dimension of the Krylov space"
	arg_type = Int64
	default = 30
end

pars = parse_args(ARGS, settings; as_symbols = true)

println("Running EKR Gilt-TNR eigenvalue computation")
println("Parameters: chi=$(pars[:chi]), gilt_eps=$(pars[:gilt_eps]), N=$(pars[:N])")

# Include and run the main eigensystem script
include("eigensystem.jl")
