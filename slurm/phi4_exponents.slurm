#!/bin/bash
#SBATCH --job-name=phi4_exp
#SBATCH --output=logs/phi4_exp_%j.out
#SBATCH --error=logs/phi4_exp_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=shared

# Load modules (if needed, though we usually use local Julia)
# module load julia

# Set up environment
export JULIA_PROJECT=ekrgilttrnr
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Parameters
MU_SQ=2.731815
CHI=32
STEPS=50
OUTPUT_DIR=ekrgilttrnr/data/phi4_exponents
mkdir -p $OUTPUT_DIR

OUTPUT_FILE=$OUTPUT_DIR/phi4_exponents_mu${MU_SQ}_chi${CHI}.dat

echo "Starting Phi4 Exponents Run"
echo "Job ID: $SLURM_JOB_ID"
echo "Mu^2: $MU_SQ"
echo "Chi: $CHI"
echo "Steps: $STEPS"
echo "Output: $OUTPUT_FILE"

# Run script
julia --project=ekrgilttrnr ekrgilttrnr/scripts/phi4_exponents.jl \
    --mu_sq $MU_SQ \
    --chi $CHI \
    --steps $STEPS \
    --output $OUTPUT_FILE

echo "Job Complete"
