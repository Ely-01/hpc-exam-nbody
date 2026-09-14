#!/bin/sh
set -eu

CONTAINER_IMAGE=${CONTAINER_IMAGE:-container/nbody_latest.sif}
CONTAINER_RUNTIME=${CONTAINER_RUNTIME:-}
OUTPUT_DIR=${OUTPUT_DIR:-results}
HOST_MPI_HOME=${HOST_MPI_HOME:-}

if [ ! -f "$CONTAINER_IMAGE" ]; then
  echo "error: container image '$CONTAINER_IMAGE' not found" >&2
  exit 1
fi

if [ -z "$CONTAINER_RUNTIME" ]; then
  if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_RUNTIME=apptainer
  elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_RUNTIME=singularity
  else
    echo "error: neither apptainer nor singularity was found" >&2
    exit 1
  fi
fi

if [ ! -x ./nbody_mpi_omp ]; then
  echo "error: ./nbody_mpi_omp not found; build it before checking ldd" >&2
  exit 1
fi

if [ -z "$HOST_MPI_HOME" ] && command -v mpicc >/dev/null 2>&1; then
  HOST_MPI_HOME=$(dirname "$(dirname "$(readlink -f "$(command -v mpicc)")")")
fi

mkdir -p "$OUTPUT_DIR"

native_out="$OUTPUT_DIR/mpi_ldd_native.txt"
container_build_out="$OUTPUT_DIR/mpi_ldd_container_build_env.txt"
container_host_out="$OUTPUT_DIR/mpi_ldd_container_host_mpi.txt"

echo "# native executable ldd -> $native_out"
ldd ./nbody_mpi_omp > "$native_out"

echo "# container build-time MPI ldd -> $container_build_out"
"$CONTAINER_RUNTIME" exec \
  --bind "$PWD:$PWD" \
  --pwd "$PWD" \
  "$CONTAINER_IMAGE" \
  ldd ./nbody_mpi_omp > "$container_build_out"

if [ -z "$HOST_MPI_HOME" ] || [ ! -d "$HOST_MPI_HOME" ]; then
  echo "warning: HOST_MPI_HOME could not be inferred; skipping host-MPI-bound ldd" >&2
  echo "set HOST_MPI_HOME to the loaded MPI prefix, for example /opt/programs/openMPI/4.1.6" >&2
else
  echo "# host MPI prefix: $HOST_MPI_HOME"
  echo "# container runtime host MPI ldd -> $container_host_out"
  APPTAINERENV_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} \
  APPTAINERENV_PATH=${PATH:-} \
  SINGULARITYENV_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} \
  SINGULARITYENV_PATH=${PATH:-} \
  "$CONTAINER_RUNTIME" exec \
    --bind "$PWD:$PWD" \
    --bind "$HOST_MPI_HOME:$HOST_MPI_HOME" \
    --pwd "$PWD" \
    "$CONTAINER_IMAGE" \
    ldd ./nbody_mpi_omp > "$container_host_out"
fi

echo "# MPI-related ldd lines"
for file in "$native_out" "$container_build_out" "$container_host_out"; do
  if [ -f "$file" ]; then
    echo "# $file"
    grep -Ei 'mpi|open-rte|open-pal|pmix|ucx|ofi|fabric|psm|hwloc' "$file" || true
  fi
done
