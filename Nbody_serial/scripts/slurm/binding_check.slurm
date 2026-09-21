#!/bin/sh
#SBATCH --job-name=nbody-bind
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=results/slurm_binding_check_%j.out
#SBATCH --error=results/slurm_binding_check_%j.err

set -eu

mkdir -p results

if [ -n "${MODULES:-}" ]; then
  # shellcheck disable=SC2086
  module load $MODULES
fi

CONFIG=${CONFIG:-"${SLURM_NTASKS}x${SLURM_CPUS_PER_TASK}"}
RANKS=${CONFIG%x*}
THREADS=${CONFIG#*x}

export OMP_NUM_THREADS=${THREADS}
export OMP_PLACES=${OMP_PLACES:-cores}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}

echo "# binding check"
echo "# config=${CONFIG}"
echo "# ranks=${RANKS}"
echo "# threads=${THREADS}"
echo "# node_list=${SLURM_NODELIST:-unknown}"
echo "# omp_num_threads=${OMP_NUM_THREADS}"
echo "# omp_places=${OMP_PLACES}"
echo "# omp_proc_bind=${OMP_PROC_BIND}"
echo

if command -v numactl >/dev/null 2>&1; then
  echo "# numactl -H"
  numactl -H
  echo
fi

echo "# rank affinity"
srun -n "${RANKS}" -c "${THREADS}" sh -c '
rank=${SLURM_PROCID:-unknown}
local_rank=${SLURM_LOCALID:-unknown}
host=$(hostname)
affinity=$(taskset -pc $$ 2>&1 | sed "s/^pid [0-9]*'\''s current affinity list: //")
printf "rank=%s local_rank=%s host=%s cpus=%s OMP_NUM_THREADS=%s OMP_PLACES=%s OMP_PROC_BIND=%s\n" \
  "$rank" "$local_rank" "$host" "$affinity" \
  "${OMP_NUM_THREADS:-unset}" "${OMP_PLACES:-unset}" "${OMP_PROC_BIND:-unset}"
'
