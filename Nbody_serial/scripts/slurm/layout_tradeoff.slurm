#!/bin/sh
#SBATCH --job-name=nbody-layout
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --output=results/slurm_layout_tradeoff_%j.out
#SBATCH --error=results/slurm_layout_tradeoff_%j.err

set -eu

mkdir -p results

# Example:
#   sbatch -A dssc -p GENOA --ntasks=1 --cpus-per-task=8 \
#     --export=ALL,MODULES="openMPI/4.1.6",THREADS="1 2 4 8",N=8192,REPEATS=5 \
#     scripts/slurm/layout_tradeoff.slurm
if [ -n "${MODULES:-}" ]; then
  # shellcheck disable=SC2086
  module load $MODULES
fi

sh scripts/benchmark/benchmark_layout_tradeoff.sh
