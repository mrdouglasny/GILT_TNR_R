#!/bin/bash
# Wrapper script to run EKR eigensystem computation with correct working directory

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default parameters
CHI=30
GILT_EPS=6e-6
N=10

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --chi)
            CHI="$2"
            shift 2
            ;;
        --gilt_eps)
            GILT_EPS="$2"
            shift 2
            ;;
        --N)
            N="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Running EKR Gilt-TNR eigenvalue computation"
echo "Parameters: chi=$CHI, gilt_eps=$GILT_EPS, N=$N"
echo "Working directory: $SCRIPT_DIR/src"

# Change to src directory and run
cd "$SCRIPT_DIR/src"
julia --project=.. -e "
# Load modules
include(\"Tools.jl\")
include(\"GaugeFixing.jl\")
include(\"KrylovTechnical.jl\")

# Set up parameters
chi = $CHI
gilt_eps = $GILT_EPS
N = $N
cg_eps = 1e-10
rotate = false
ord = 2
stp = 1e-4
verbosity = 1
Z2_odd_sector = true
freeze_R = false
krylovdim = 30
relT = 0.0
Jratio = 1.0
number_of_initial_steps = 23

gilt_pars = Dict(
    \"gilt_eps\" => gilt_eps,
    \"cg_chis\" => collect(1:chi),
    \"cg_eps\" => cg_eps,
    \"verbosity\" => 0,
    \"rotate\" => rotate,
)

println(\"Finding critical temperature...\")
search_tol = 1.0e-10
relT_low, relT_high, relT = find_critical_temperature(chi, gilt_pars, search_tol, Jratio)
println(\"Critical relT = \$relT\")

println(\"Generating initial tensor...\")
A, recursion_depths = initial_tensor(relT, Jratio, number_of_initial_steps, gilt_pars)

println(\"Computing eigenvalues...\")
# Note: eigensystem computation would go here
# This requires the full eigensystem.jl logic
println(\"Setup complete. Run the full eigensystem.jl for actual computation.\")
"
