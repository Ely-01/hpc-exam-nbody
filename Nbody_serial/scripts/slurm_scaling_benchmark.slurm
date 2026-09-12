#!/bin/sh
#SBATCH --job-name=nbody-scaling
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm_scaling_benchmark_%j.out
#SBATCH --error=results/slurm_scaling_benchmark_%j.err

set -eu

mkdir -p results

# Optional: pass modules at submission time, for example:
#   sbatch --export=ALL,MODULES="openMPI/4.1.6" scripts/slurm_scaling_benchmark.slurm
if [ -n "${MODULES:-}" ]; then
  # shellcheck disable=SC2086
  module load $MODULES
fi

scripts/benchmark_scaling.sh
