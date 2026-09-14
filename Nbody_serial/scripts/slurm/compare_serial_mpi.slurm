#!/bin/sh
#SBATCH --job-name=nbody-compare
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm_compare_serial_mpi_%j.out
#SBATCH --error=results/slurm_compare_serial_mpi_%j.err

set -eu

mkdir -p results

# Optional: pass modules at submission time, for example:
#   sbatch --export=ALL,MODULES="openMPI/4.1.6" scripts/slurm/compare_serial_mpi.slurm
if [ -n "${MODULES:-}" ]; then
  # shellcheck disable=SC2086
  module load $MODULES
fi

RANKS=${RANKS:-"1 2 4"} \
OMP_THREADS=${SLURM_CPUS_PER_TASK:-1} \
sh scripts/smoke/compare_serial_mpi.sh
