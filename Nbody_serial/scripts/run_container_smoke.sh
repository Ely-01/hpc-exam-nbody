#!/bin/sh
set -eu

CONTAINER_IMAGE=${CONTAINER_IMAGE:-container/nbody_latest.sif}
N=${N:-128}
NSTEPS=${NSTEPS:-5}
DT=${DT:-1e-4}
EPS=${EPS:-0.05}
MASS=${MASS:-1.0}
SEED=${SEED:-123}
MPI_RANKS=${MPI_RANKS:-${SLURM_NTASKS:-4}}
OMP_THREADS=${OMP_THREADS:-${SLURM_CPUS_PER_TASK:-1}}
OUTPUT_DIR=${OUTPUT_DIR:-results}
INPUT=${INPUT:-$OUTPUT_DIR/plummer_container_${N}.bin}
BUILD_CFLAGS=${BUILD_CFLAGS:-"-O3 -march=x86-64-v3 -ffp-contract=fast -Wall -Wextra -Wpedantic"}
HOST_MPI_HOME=${HOST_MPI_HOME:-}

if [ ! -f "$CONTAINER_IMAGE" ]; then
  echo "error: container image '$CONTAINER_IMAGE' not found" >&2
  echo "build it with: apptainer build $CONTAINER_IMAGE container/nbody.def" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

container_exec () {
  apptainer exec \
    --bind "$PWD:$PWD" \
    --pwd "$PWD" \
    "$CONTAINER_IMAGE" \
    "$@"
}

if [ -z "$HOST_MPI_HOME" ] && command -v mpicc >/dev/null 2>&1; then
  HOST_MPI_HOME=$(dirname "$(dirname "$(readlink -f "$(command -v mpicc)")")")
fi

runtime_container_exec () {
  if [ -n "$HOST_MPI_HOME" ] && [ -d "$HOST_MPI_HOME" ]; then
    APPTAINERENV_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} \
    APPTAINERENV_PATH=${PATH:-} \
    apptainer exec \
      --bind "$PWD:$PWD" \
      --bind "$HOST_MPI_HOME:$HOST_MPI_HOME" \
      --pwd "$PWD" \
      "$CONTAINER_IMAGE" \
      "$@"
  else
    apptainer exec \
      --bind "$PWD:$PWD" \
      --pwd "$PWD" \
      "$CONTAINER_IMAGE" \
      "$@"
  fi
}

echo "# container smoke"
echo "# image=$CONTAINER_IMAGE"
echo "# ranks=$MPI_RANKS omp_threads=$OMP_THREADS"
echo "# n=$N nsteps=$NSTEPS dt=$DT eps=$EPS mass=$MASS"
echo "# build_cflags=$BUILD_CFLAGS"
echo "# host_mpi_home=${HOST_MPI_HOME:-not-set}"

container_exec make clean
container_exec make OPENMP=1 PRECISION=double CFLAGS="$BUILD_CFLAGS"
container_exec make mpi OPENMP=1 PRECISION=double CFLAGS="$BUILD_CFLAGS"

if [ ! -f "$INPUT" ]; then
  container_exec ./generate_ic \
    --model 0 \
    --n "$N" \
    --seed "$SEED" \
    --scale 1.0 \
    --mass "$MASS" \
    --output "$INPUT"
fi

echo "# serial container smoke"
runtime_container_exec ./nbody_direct_serial \
  --input "$INPUT" \
  --nsteps "$NSTEPS" \
  --dt "$DT" \
  --eps "$EPS" \
  --mass "$MASS" \
  --energy-every "$NSTEPS" \
  --timing \
  --quiet

echo "# mpi container smoke"
export OMP_NUM_THREADS="$OMP_THREADS"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1; then
  if [ -n "$HOST_MPI_HOME" ] && [ -d "$HOST_MPI_HOME" ]; then
    srun -n "$MPI_RANKS" -c "$OMP_THREADS" \
      env APPTAINERENV_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
        APPTAINERENV_PATH="${PATH:-}" \
        apptainer exec \
          --bind "$PWD:$PWD" \
          --bind "$HOST_MPI_HOME:$HOST_MPI_HOME" \
          --pwd "$PWD" \
          "$CONTAINER_IMAGE" \
          ./nbody_mpi_omp \
            --input "$INPUT" \
            --nsteps "$NSTEPS" \
            --dt "$DT" \
            --eps "$EPS" \
            --mass "$MASS" \
            --energy-every "$NSTEPS" \
            --timing \
            --quiet
  else
    srun -n "$MPI_RANKS" -c "$OMP_THREADS" \
      apptainer exec \
        --bind "$PWD:$PWD" \
        --pwd "$PWD" \
        "$CONTAINER_IMAGE" \
        ./nbody_mpi_omp \
          --input "$INPUT" \
          --nsteps "$NSTEPS" \
          --dt "$DT" \
          --eps "$EPS" \
          --mass "$MASS" \
          --energy-every "$NSTEPS" \
          --timing \
          --quiet
  fi
elif command -v mpirun >/dev/null 2>&1; then
  if [ -n "$HOST_MPI_HOME" ] && [ -d "$HOST_MPI_HOME" ]; then
    mpirun -np "$MPI_RANKS" \
      env APPTAINERENV_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
        APPTAINERENV_PATH="${PATH:-}" \
        apptainer exec \
          --bind "$PWD:$PWD" \
          --bind "$HOST_MPI_HOME:$HOST_MPI_HOME" \
          --pwd "$PWD" \
          "$CONTAINER_IMAGE" \
          ./nbody_mpi_omp \
            --input "$INPUT" \
            --nsteps "$NSTEPS" \
            --dt "$DT" \
            --eps "$EPS" \
            --mass "$MASS" \
            --energy-every "$NSTEPS" \
            --timing \
            --quiet
  else
    mpirun -np "$MPI_RANKS" \
      apptainer exec \
        --bind "$PWD:$PWD" \
        --pwd "$PWD" \
        "$CONTAINER_IMAGE" \
        ./nbody_mpi_omp \
          --input "$INPUT" \
          --nsteps "$NSTEPS" \
          --dt "$DT" \
          --eps "$EPS" \
          --mass "$MASS" \
          --energy-every "$NSTEPS" \
          --timing \
          --quiet
  fi
else
  echo "error: neither srun nor mpirun was found" >&2
  exit 1
fi
