#!/bin/bash
#SBATCH --job-name=phi4_mu_scan
#SBATCH --output=ekrgilttrnr/logs/slurm/phi4_mu_scan_%x_%j.out
#SBATCH --error=ekrgilttrnr/logs/slurm/phi4_mu_scan_%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=sapphire,shared

# Script to run phi4_mu_scan.jl on SLURM cluster
# Scans a single mu_sq value with chi=30, gilt_eps=1e-6
#
# Usage: sbatch ekrgilttrnr/slurm/phi4_mu_scan.slurm <config_name>
#   config_name: e.g., mu_m1.34 for mu_sq = -1.34

# Get config name from command line
CONFIG_NAME=${1:-mu_m1.34}

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
if [ -f "ekrgilttrnr/scripts/phi4_mu_scan.jl" ]; then
    PROJECT_ROOT="$(pwd)"
elif [ -f "../ekrgilttrnr/scripts/phi4_mu_scan.jl" ]; then
    cd ..
    PROJECT_ROOT="$(pwd)"
elif [ -f "../../ekrgilttrnr/scripts/phi4_mu_scan.jl" ]; then
    cd ../..
    PROJECT_ROOT="$(pwd)"
else
    cd $SLURM_SUBMIT_DIR
    PROJECT_ROOT="$(pwd)"
fi

echo "Working directory: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Ensure log and data directories exist
mkdir -p ekrgilttrnr/logs/slurm
mkdir -p ekrgilttrnr/logs/phi4_mu_scan
mkdir -p ekrgilttrnr/data/phi4_mu_scan

# Check Julia version
echo "Julia version:"
julia --version

# Check if config exists
CONFIG_FILE="ekrgilttrnr/configs/phi4_mu_scan/${CONFIG_NAME}.toml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Using config: $CONFIG_FILE"
echo ""

# Run the mu_sq scan
echo "=========================================="
echo "Starting phi4_mu_scan.jl..."
echo "=========================================="

julia --project=ekrgilttrnr -t$SLURM_CPUS_PER_TASK ekrgilttrnr/scripts/phi4_mu_scan.jl $CONFIG_NAME

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

# Check if output was created
OUTPUT_DATA="ekrgilttrnr/data/phi4_mu_scan/${CONFIG_NAME}.csv"
if [ -f "$OUTPUT_DATA" ]; then
    echo "Output created successfully: $OUTPUT_DATA"
    echo "Size: $(ls -lh $OUTPUT_DATA | awk '{print $5}')"
else
    echo "WARNING: Output file not found: $OUTPUT_DATA"
fi

exit $EXIT_CODE
