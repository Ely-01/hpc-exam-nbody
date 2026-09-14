#!/bin/sh
#SBATCH --job-name=nbody-scaling
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm_scaling_benchmark_%j.out
#SBATCH --error=results/slurm_scaling_benchmark_%j.err

set -eu

mkdir -p results

# Request the actual MPI/OpenMP allocation at submission time.  This script runs
# all configurations listed in CONFIGS, so the allocation must cover the largest
# ranks x threads case:
#
# MPI-only example:
#   sbatch -A dssc -p GENOA --ntasks=32 --cpus-per-task=1 \
#     --export=ALL,MODULES="openMPI/4.1.6",MODE=strong,CONFIGS="1x1 2x1 4x1 8x1 16x1 32x1" \
#     scripts/slurm/scaling_benchmark.slurm
#
# Hybrid example:
#   sbatch -A dssc -p GENOA --ntasks=8 --cpus-per-task=4 \
#     --export=ALL,MODULES="openMPI/4.1.6",MODE=strong,CONFIGS="1x4 2x4 4x4 8x4" \
#     scripts/slurm/scaling_benchmark.slurm
#
# Optional: pass modules at submission time, for example:
#   sbatch --export=ALL,MODULES="openMPI/4.1.6" scripts/slurm/scaling_benchmark.slurm
if [ -n "${MODULES:-}" ]; then
  # shellcheck disable=SC2086
  module load $MODULES
fi

sh scripts/benchmark/benchmark_scaling.sh
