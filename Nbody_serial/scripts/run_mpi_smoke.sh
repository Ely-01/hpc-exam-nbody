#!/bin/sh
set -eu

N=${N:-128}
NSTEPS=${NSTEPS:-5}
DT=${DT:-1e-4}
EPS=${EPS:-0.05}
MASS=${MASS:-1.0}
SEED=${SEED:-123}
ENERGY_EVERY=${ENERGY_EVERY:-1}
MPI_RANKS=${MPI_RANKS:-${SLURM_NTASKS:-4}}
OMP_THREADS=${OMP_THREADS:-${SLURM_CPUS_PER_TASK:-1}}
OUTPUT_DIR=${OUTPUT_DIR:-results}
INPUT=${INPUT:-$OUTPUT_DIR/plummer_${N}.bin}
LOG=${LOG:-$OUTPUT_DIR/mpi_smoke_$(date +%Y%m%d_%H%M%S).log}

mkdir -p "$OUTPUT_DIR"

make mpi OPENMP=1 PRECISION=double

if [ ! -f "$INPUT" ]; then
  ./generate_ic \
    --model 0 \
    --n "$N" \
    --seed "$SEED" \
    --scale 1.0 \
    --mass "$MASS" \
    --output "$INPUT"
fi

export OMP_NUM_THREADS="$OMP_THREADS"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

echo "# mpi smoke"
echo "# input=$INPUT"
echo "# ranks=$MPI_RANKS omp_threads=$OMP_THREADS"
echo "# n=$N nsteps=$NSTEPS dt=$DT eps=$EPS mass=$MASS"
echo "# log=$LOG"

if command -v srun >/dev/null 2>&1; then
  srun -n "$MPI_RANKS" -c "$OMP_THREADS" \
    ./nbody_mpi_omp \
      --input "$INPUT" \
      --nsteps "$NSTEPS" \
      --dt "$DT" \
      --eps "$EPS" \
      --mass "$MASS" \
      --energy-every "$ENERGY_EVERY" \
      --timing | tee "$LOG"
elif command -v mpirun >/dev/null 2>&1; then
  mpirun -np "$MPI_RANKS" \
    ./nbody_mpi_omp \
      --input "$INPUT" \
      --nsteps "$NSTEPS" \
      --dt "$DT" \
      --eps "$EPS" \
      --mass "$MASS" \
      --energy-every "$ENERGY_EVERY" \
      --timing | tee "$LOG"
else
  echo "error: neither srun nor mpirun was found" >&2
  exit 1
fi

