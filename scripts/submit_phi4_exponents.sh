#!/bin/bash
# Submit the Phi4 exponents job to the cluster

# Ensure logs directory exists
mkdir -p logs

# Submit
sbatch ekrgilttrnr/slurm/phi4_exponents.slurm
