#!/bin/sh
#SBATCH --job-name=nbody-valid
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm_validation_%j.out
#SBATCH --error=results/slurm_validation_%j.err

set -eu

mkdir -p results

# Submit with the desired allocation, for example:
#   sbatch -A dssc -p GENOA --ntasks=1 --cpus-per-task=8 \
#     --export=ALL,MODULES="openMPI/4.1.6",THREADS=8 \
#     scripts/slurm/validation.slurm
if [ -n "${MODULES:-}" ]; then
  # shellcheck disable=SC2086
  module load $MODULES
fi

sh scripts/benchmark/benchmark_validation.sh
