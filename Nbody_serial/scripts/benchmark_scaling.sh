#!/bin/sh
set -eu

MODE=${MODE:-strong}
N=${N:-32768}
NLOCAL=${NLOCAL:-4096}
NSTEPS=${NSTEPS:-20}
DT=${DT:-1e-4}
EPS=${EPS:-0.05}
MASS=${MASS:-1.0}
SEED=${SEED:-123}
ENERGY_EVERY=${ENERGY_EVERY:-$NSTEPS}
REPEATS=${REPEATS:-5}
CONFIGS=${CONFIGS:-"1x1 2x1 4x1 8x1"}
OUTPUT_DIR=${OUTPUT_DIR:-results}
CSV=${CSV:-$OUTPUT_DIR/${MODE}_scaling_$(date +%Y%m%d_%H%M%S).csv}

mkdir -p "$OUTPUT_DIR"

make OPENMP=1 PRECISION=double
make mpi OPENMP=1 PRECISION=double

case "$MODE" in
  strong|weak)
    ;;
  *)
    echo "error: MODE must be either strong or weak" >&2
    exit 1
    ;;
esac

printf '%s\n' \
  "mode,n,nlocal,nsteps,dt,eps,mass,ranks,threads,total_workers,repeat,total_seconds,force_seconds,communication_seconds,integration_seconds,energy_seconds,initial_acceleration_seconds,max_relative_energy_drift,status" \
  > "$CSV"

max_ranks=0
max_threads=0
max_workers=0
for config in $CONFIGS; do
  ranks=${config%x*}
  threads=${config#*x}

  if [ "$ranks" = "$config" ] || [ -z "$ranks" ] || [ -z "$threads" ]; then
    echo "error: invalid CONFIGS entry '$config', expected ranksxthreads" >&2
    exit 1
  fi

  workers=$((ranks * threads))
  if [ "$ranks" -gt "$max_ranks" ]; then
    max_ranks=$ranks
  fi
  if [ "$threads" -gt "$max_threads" ]; then
    max_threads=$threads
  fi
  if [ "$workers" -gt "$max_workers" ]; then
    max_workers=$workers
  fi
done

if [ -n "${SLURM_NTASKS:-}" ] && [ "$max_ranks" -gt "$SLURM_NTASKS" ]; then
  echo "error: CONFIGS require up to $max_ranks MPI ranks, but SLURM_NTASKS=$SLURM_NTASKS" >&2
  exit 1
fi

if [ -n "${SLURM_CPUS_PER_TASK:-}" ] && [ "$max_threads" -gt "$SLURM_CPUS_PER_TASK" ]; then
  echo "error: CONFIGS require up to $max_threads OpenMP threads per rank, but SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK" >&2
  exit 1
fi

echo "# scaling mode=$MODE configs=$CONFIGS max_ranks=$max_ranks max_threads=$max_threads max_workers=$max_workers"

extract_final_field () {
  key=$1
  awk -v key="$key" '
    $1 == "#" && $2 == "final:" {
      for (i = 1; i <= NF; ++i) {
        split($i, field, "=")
        if (field[1] == key) {
          print field[2]
          found = 1
          exit
        }
      }
    }
    END {
      if (!found)
        print "nan"
    }
  '
}

extract_timing () {
  key=$1
  awk -v key="$key" '
    $1 == "#" && $2 == "timing" && $3 == key {
      print $4
      found = 1
      exit
    }
    END {
      if (!found)
        print "0"
    }
  '
}

run_mpi () {
  ranks=$1
  threads=$2
  input=$3
  n_for_run=$4

  if [ $((n_for_run % ranks)) -ne 0 ]; then
    echo "error: N=$n_for_run must be divisible by ranks=$ranks" >&2
    exit 1
  fi

  if command -v srun >/dev/null 2>&1; then
    srun -n "$ranks" -c "$threads" \
      ./nbody_mpi_omp \
        --input "$input" \
        --nsteps "$NSTEPS" \
        --dt "$DT" \
        --eps "$EPS" \
        --mass "$MASS" \
        --energy-every "$ENERGY_EVERY" \
        --timing \
        --quiet
  elif command -v mpirun >/dev/null 2>&1; then
    mpirun -np "$ranks" \
      ./nbody_mpi_omp \
        --input "$input" \
        --nsteps "$NSTEPS" \
        --dt "$DT" \
        --eps "$EPS" \
        --mass "$MASS" \
        --energy-every "$ENERGY_EVERY" \
        --timing \
        --quiet
  else
    echo "error: neither srun nor mpirun was found" >&2
    exit 1
  fi
}

make_input () {
  n_for_input=$1
  input=$2

  if [ ! -f "$input" ]; then
    ./generate_ic \
      --model 0 \
      --n "$n_for_input" \
      --seed "$SEED" \
      --scale 1.0 \
      --mass "$MASS" \
      --output "$input"
  fi
}

for config in $CONFIGS; do
  ranks=${config%x*}
  threads=${config#*x}
  total_workers=$((ranks * threads))

  if [ "$ranks" = "$config" ] || [ -z "$ranks" ] || [ -z "$threads" ]; then
    echo "error: invalid CONFIGS entry '$config', expected ranksxthreads" >&2
    exit 1
  fi

  if [ "$MODE" = "strong" ]; then
    n_run=$N
  else
    n_run=$((NLOCAL * ranks))
  fi

  nlocal_run=$((n_run / ranks))
  input="$OUTPUT_DIR/plummer_${MODE}_${n_run}.bin"
  make_input "$n_run" "$input"

  for repeat in $(seq 1 "$REPEATS"); do
    echo "# $MODE scaling run n=$n_run ranks=$ranks threads=$threads repeat=$repeat/$REPEATS"

    output=$(
      OMP_NUM_THREADS="$threads" \
      OMP_PROC_BIND="${OMP_PROC_BIND:-close}" \
      OMP_PLACES="${OMP_PLACES:-cores}" \
      run_mpi "$ranks" "$threads" "$input" "$n_run"
    )

    printf '%s\n' "$output"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$MODE" \
      "$n_run" \
      "$nlocal_run" \
      "$NSTEPS" \
      "$DT" \
      "$EPS" \
      "$MASS" \
      "$ranks" \
      "$threads" \
      "$total_workers" \
      "$repeat" \
      "$(printf '%s\n' "$output" | extract_timing total_seconds)" \
      "$(printf '%s\n' "$output" | extract_timing force_seconds)" \
      "$(printf '%s\n' "$output" | extract_timing communication_seconds)" \
      "$(printf '%s\n' "$output" | extract_timing integration_seconds)" \
      "$(printf '%s\n' "$output" | extract_timing energy_seconds)" \
      "$(printf '%s\n' "$output" | extract_timing initial_acceleration_seconds)" \
      "$(printf '%s\n' "$output" | extract_final_field max_relative_energy_drift)" \
      "$(printf '%s\n' "$output" | extract_final_field status)" \
      >> "$CSV"
  done
done

echo "# wrote $CSV"
