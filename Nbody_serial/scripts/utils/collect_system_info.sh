#!/bin/sh
set -eu

section() {
  printf '\n## %s\n' "$1"
}

run_optional() {
  label=$1
  shift

  section "$label"
  if command -v "$1" >/dev/null 2>&1; then
    "$@" || true
  else
    printf 'not available: %s\n' "$1"
  fi
}

section "Job context"
printf 'date: %s\n' "$(date -Is 2>/dev/null || date)"
printf 'hostname: %s\n' "$(hostname)"
printf 'pwd: %s\n' "$PWD"
printf 'user: %s\n' "${USER:-unknown}"
printf 'slurm_job_id: %s\n' "${SLURM_JOB_ID:-not-set}"
printf 'slurm_partition: %s\n' "${SLURM_JOB_PARTITION:-not-set}"
printf 'slurm_nodes: %s\n' "${SLURM_JOB_NODELIST:-not-set}"
printf 'slurm_ntasks: %s\n' "${SLURM_NTASKS:-not-set}"
printf 'slurm_cpus_per_task: %s\n' "${SLURM_CPUS_PER_TASK:-not-set}"

section "Thread and binding environment"
printf 'OMP_NUM_THREADS: %s\n' "${OMP_NUM_THREADS:-not-set}"
printf 'OMP_PLACES: %s\n' "${OMP_PLACES:-not-set}"
printf 'OMP_PROC_BIND: %s\n' "${OMP_PROC_BIND:-not-set}"
printf 'OMPI_MCA_pml: %s\n' "${OMPI_MCA_pml:-not-set}"
printf 'OMPI_MCA_btl: %s\n' "${OMPI_MCA_btl:-not-set}"
printf 'OMPI_MCA_osc: %s\n' "${OMPI_MCA_osc:-not-set}"

run_optional "Kernel" uname -a
run_optional "CPU topology" lscpu
run_optional "NUMA topology" numactl -H
run_optional "Memory" free -h

section "Loaded modules"
if command -v module >/dev/null 2>&1; then
  module list 2>&1 || true
else
  printf 'not available: module\n'
fi

section "Compiler and MPI"
for tool in cc gcc mpicc mpirun mpiexec singularity apptainer; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf '%s: %s\n' "$tool" "$(command -v "$tool")"
    "$tool" --version 2>&1 | sed -n '1,3p' || true
  else
    printf '%s: not available\n' "$tool"
  fi
  printf '\n'
done

run_optional "C library" ldd --version

section "MPI executable linkage"
if [ -x ./nbody_mpi_omp ]; then
  ldd ./nbody_mpi_omp 2>&1 | grep -Ei 'mpi|open-rte|open-pal|pmix|ucx|ofi|fabric|psm|hwloc|gomp|omp' || true
else
  printf './nbody_mpi_omp not built in this directory\n'
fi
