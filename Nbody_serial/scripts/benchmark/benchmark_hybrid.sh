#!/bin/sh
set -eu

N=${N:-4096}
NSTEPS=${NSTEPS:-10}
DT=${DT:-1e-4}
EPS=${EPS:-0.05}
MASS=${MASS:-1.0}
SEED=${SEED:-123}
ENERGY_EVERY=${ENERGY_EVERY:-$NSTEPS}
REPEATS=${REPEATS:-5}
CONFIGS=${CONFIGS:-"1x4 2x2 4x1"}
RING_MODE=${RING_MODE:-blocking}
OUTPUT_DIR=${OUTPUT_DIR:-results}
INPUT=${INPUT:-$OUTPUT_DIR/plummer_${N}.bin}
CSV=${CSV:-$OUTPUT_DIR/hybrid_benchmark_${RING_MODE}_$(date +%Y%m%d_%H%M%S).csv}

mkdir -p "$OUTPUT_DIR"

make OPENMP=1 PRECISION=double
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

printf '%s\n' \
  "backend,n,nsteps,dt,eps,mass,ranks,threads,total_workers,repeat,total_seconds,force_seconds,communication_seconds,integration_seconds,energy_seconds,initial_acceleration_seconds,max_relative_energy_drift,status" \
  > "$CSV"

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

  if [ $((N % ranks)) -ne 0 ]; then
    echo "error: N=$N must be divisible by ranks=$ranks" >&2
    exit 1
  fi

  if command -v srun >/dev/null 2>&1; then
    srun -n "$ranks" -c "$threads" \
      ./nbody_mpi_omp \
        --input "$INPUT" \
        --nsteps "$NSTEPS" \
        --dt "$DT" \
        --eps "$EPS" \
        --mass "$MASS" \
        --energy-every "$ENERGY_EVERY" \
        --ring-mode "$RING_MODE" \
        --timing \
        --quiet
  elif command -v mpirun >/dev/null 2>&1; then
    mpirun -np "$ranks" \
      ./nbody_mpi_omp \
        --input "$INPUT" \
        --nsteps "$NSTEPS" \
        --dt "$DT" \
        --eps "$EPS" \
        --mass "$MASS" \
        --energy-every "$ENERGY_EVERY" \
        --ring-mode "$RING_MODE" \
        --timing \
        --quiet
  else
    echo "error: neither srun nor mpirun was found" >&2
    exit 1
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

  for repeat in $(seq 1 "$REPEATS"); do
    echo "# hybrid run ring_mode=$RING_MODE ranks=$ranks threads=$threads repeat=$repeat/$REPEATS"

    output=$(
      OMP_NUM_THREADS="$threads" \
      OMP_PROC_BIND="${OMP_PROC_BIND:-close}" \
      OMP_PLACES="${OMP_PLACES:-cores}" \
      run_mpi "$ranks" "$threads"
    )

    printf '%s\n' "$output"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "mpi_kdk_${RING_MODE}" \
      "$N" \
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
