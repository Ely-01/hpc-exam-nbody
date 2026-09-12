#!/bin/sh
#SBATCH --job-name=nbody-mpi-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm_mpi_smoke_%j.out
#SBATCH --error=results/slurm_mpi_smoke_%j.err

set -eu

mkdir -p results

# Optional: pass modules at submission time, for example:
#   sbatch --export=ALL,MODULES="gcc openmpi" scripts/slurm_mpi_smoke.slurm
if [ -n "${MODULES:-}" ]; then
  # shellcheck disable=SC2086
  module load $MODULES
fi

MPI_RANKS=${SLURM_NTASKS:-4} \
OMP_THREADS=${SLURM_CPUS_PER_TASK:-1} \
scripts/run_mpi_smoke.sh
