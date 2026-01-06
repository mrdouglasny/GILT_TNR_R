#!/bin/bash
#SBATCH --job-name=phi4_mu_scan
#SBATCH --output=ekrgilttrnr/logs/phi4_mu_scan/%A_%a.out
#SBATCH --error=ekrgilttrnr/logs/phi4_mu_scan/%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=sapphire,shared
#SBATCH --array=0-12

# φ⁴ μ² scan at χ=30, gilt_eps=1e-8 (gentle GILT)
# Fine scan from -1.40 to -1.28 in steps of 0.01
# 13 tasks total

# Map task ID to μ² value
MU_VALUES=(-1.40 -1.39 -1.38 -1.37 -1.36 -1.35 -1.34 -1.33 -1.32 -1.31 -1.30 -1.29 -1.28)

MU_SQ=${MU_VALUES[$SLURM_ARRAY_TASK_ID]}
CONFIG_NAME="mu_m${MU_SQ#-}"  # Remove leading minus, add m prefix

echo "Task $SLURM_ARRAY_TASK_ID: μ² = $MU_SQ, config = $CONFIG_NAME"
echo "Parameters: χ=30, gilt_eps=1e-6, max_steps=40"

# Load Julia module
module load julia/1.10.4 2>/dev/null || true

# Run the scan
cd $SLURM_SUBMIT_DIR
julia --project=ekrgilttrnr -t4 ekrgilttrnr/scripts/phi4_mu_scan.jl $CONFIG_NAME

echo "Task $SLURM_ARRAY_TASK_ID completed"
