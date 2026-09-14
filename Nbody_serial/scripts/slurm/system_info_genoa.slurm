#!/bin/sh
#SBATCH --job-name=nbody-sysinfo
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=results/system_info_genoa_%j.txt
#SBATCH --error=results/system_info_genoa_%j.err

set -eu

mkdir -p results

if [ -n "${MODULES:-}" ]; then
  # shellcheck disable=SC2086
  module load $MODULES
fi

if command -v make >/dev/null 2>&1; then
  make mpi OPENMP=1 PRECISION=double >/dev/null 2>&1 || true
fi

sh scripts/utils/collect_system_info.sh
