#!/bin/sh
set -eu

N=${N:-4096}
NSTEPS=${NSTEPS:-10}
DT=${DT:-1e-4}
EPS=${EPS:-0.05}
MASS=${MASS:-1.0}
SEED=${SEED:-123}
INTEGRATOR=${INTEGRATOR:-kdk}
ENERGY_EVERY=${ENERGY_EVERY:-$NSTEPS}
THREADS=${THREADS:-"1 2 4 8"}
REPEATS=${REPEATS:-5}
OUTPUT_DIR=${OUTPUT_DIR:-results}
INPUT=${INPUT:-$OUTPUT_DIR/plummer_${N}.bin}
CSV=${CSV:-$OUTPUT_DIR/openmp_benchmark_$(date +%Y%m%d_%H%M%S).csv}
OMP_BIND=${OMP_PROC_BIND:-close}

mkdir -p "$OUTPUT_DIR"

make OPENMP=1

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
  "n,nsteps,dt,eps,mass,integrator,threads,repeat,total_seconds,force_seconds,integration_seconds,energy_seconds,initial_acceleration_seconds,max_relative_energy_drift,status" \
  > "$CSV"

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
        print "nan"
    }
  '
}

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

for threads in $THREADS; do
  for repeat in $(seq 1 "$REPEATS"); do
    echo "# run threads=$threads repeat=$repeat/$REPEATS"

    output=$(
      OMP_NUM_THREADS="$threads" \
      OMP_PROC_BIND="$OMP_BIND" \
      ./nbody_direct_serial \
        --input "$INPUT" \
        --nsteps "$NSTEPS" \
        --dt "$DT" \
        --eps "$EPS" \
        --mass "$MASS" \
        --energy-every "$ENERGY_EVERY" \
        --integrator "$INTEGRATOR" \
        --timing \
        --quiet
    )

    openmp_status=$(printf '%s\n' "$output" | awk '$1 == "#" && $2 == "timing" && $3 == "openmp" { print $4; exit }')
    if [ "$openmp_status" != "enabled" ]; then
      printf '%s\n' "$output" >&2
      echo "error: OpenMP is not enabled in nbody_direct_serial" >&2
      exit 1
    fi

    total_seconds=$(printf '%s\n' "$output" | extract_timing total_seconds)
    force_seconds=$(printf '%s\n' "$output" | extract_timing force_seconds)
    integration_seconds=$(printf '%s\n' "$output" | extract_timing integration_seconds)
    energy_seconds=$(printf '%s\n' "$output" | extract_timing energy_seconds)
    initial_acceleration_seconds=$(printf '%s\n' "$output" | extract_timing initial_acceleration_seconds)
    drift=$(printf '%s\n' "$output" | extract_final_field max_relative_energy_drift)
    status=$(printf '%s\n' "$output" | extract_final_field status)

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$N" \
      "$NSTEPS" \
      "$DT" \
      "$EPS" \
      "$MASS" \
      "$INTEGRATOR" \
      "$threads" \
      "$repeat" \
      "$total_seconds" \
      "$force_seconds" \
      "$integration_seconds" \
      "$energy_seconds" \
      "$initial_acceleration_seconds" \
      "$drift" \
      "$status" \
      >> "$CSV"
  done
done

echo "# wrote $CSV"
