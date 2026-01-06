#!/bin/bash
#SBATCH --job-name=phi4_eigen
#SBATCH --output=ekrgilttrnr/logs/slurm/phi4_eigensystem_%x_%j.out
#SBATCH --error=ekrgilttrnr/logs/slurm/phi4_eigensystem_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=shared

# Script to run phi4_eigensystem.jl on SLURM cluster
# Computes RG Jacobian eigenvalues for phi4 model at criticality
#
# Usage: sbatch ekrgilttrnr/slurm/phi4_eigensystem.slurm [config_name]
#   config_name: quick (~5 min) or default (~30 min)

# Get config name from command line or use default
CONFIG_NAME=${1:-default}

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Config: $CONFIG_NAME"
echo "Started: $(date)"
echo "=========================================="

# Set number of threads for Julia
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Move to project root
if [ -f "ekrgilttrnr/scripts/phi4_eigensystem.jl" ]; then
    PROJECT_ROOT="$(pwd)"
elif [ -f "../ekrgilttrnr/scripts/phi4_eigensystem.jl" ]; then
    cd ..
    PROJECT_ROOT="$(pwd)"
elif [ -f "../../ekrgilttrnr/scripts/phi4_eigensystem.jl" ]; then
    cd ../..
    PROJECT_ROOT="$(pwd)"
else
    cd $SLURM_SUBMIT_DIR
    PROJECT_ROOT="$(pwd)"
fi

echo "Working directory: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Ensure log directory exists
mkdir -p ekrgilttrnr/logs/slurm
mkdir -p ekrgilttrnr/logs/phi4_eigensystem
mkdir -p ekrgilttrnr/data/phi4_eigensystem

# Check Julia version
echo "Julia version:"
julia --version

# Check if config exists
CONFIG_FILE="ekrgilttrnr/configs/phi4_eigensystem/${CONFIG_NAME}.toml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Using config: $CONFIG_FILE"
echo ""

# Run the eigensystem computation
echo "=========================================="
echo "Starting phi4_eigensystem.jl..."
echo "=========================================="

julia --project=ekrgilttrnr ekrgilttrnr/scripts/phi4_eigensystem.jl $CONFIG_NAME

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

# Check if output was created
OUTPUT_DATA="ekrgilttrnr/data/phi4_eigensystem/${CONFIG_NAME}.data"
if [ -f "$OUTPUT_DATA" ]; then
    echo "Output created successfully: $OUTPUT_DATA"
    echo "Size: $(ls -lh $OUTPUT_DATA | awk '{print $5}')"
else
    echo "WARNING: Output file not found: $OUTPUT_DATA"
fi

exit $EXIT_CODE
