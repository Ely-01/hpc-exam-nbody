#!/bin/sh
#SBATCH --job-name=nbody-hybrid
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=results/slurm_hybrid_benchmark_%j.out
#SBATCH --error=results/slurm_hybrid_benchmark_%j.err

set -eu

mkdir -p results

# Optional: pass modules at submission time, for example:
#   sbatch --export=ALL,MODULES="openMPI/4.1.6" scripts/slurm_hybrid_benchmark.slurm
if [ -n "${MODULES:-}" ]; then
  # shellcheck disable=SC2086
  module load $MODULES
fi

# Keep defaults modest.  Override at submission time for larger runs, e.g.:
#   sbatch --export=ALL,MODULES="openMPI/4.1.6",N=8192,REPEATS=5,CONFIGS="1x8 2x4 4x2 8x1" ...
CONFIGS=${CONFIGS:-"1x4 2x2 4x1"} \
REPEATS=${REPEATS:-3} \
scripts/benchmark_hybrid.sh
