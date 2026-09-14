#!/bin/sh
#SBATCH --job-name=nbody-container-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm_container_smoke_%j.out
#SBATCH --error=results/slurm_container_smoke_%j.err

set -eu

mkdir -p results

if [ -n "${MODULES:-}" ]; then
  # shellcheck disable=SC2086
  module load $MODULES
fi

MPI_RANKS=${SLURM_NTASKS:-4} \
OMP_THREADS=${SLURM_CPUS_PER_TASK:-1} \
sh scripts/smoke/run_container_smoke.sh
