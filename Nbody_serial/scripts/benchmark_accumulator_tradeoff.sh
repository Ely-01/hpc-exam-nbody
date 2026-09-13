#!/bin/sh
set -eu

N=${N:-4096}
NSTEPS=${NSTEPS:-10}
DT=${DT:-1e-4}
EPS=${EPS:-0.05}
MASS=${MASS:-1.0}
SEED=${SEED:-123}
ENERGY_EVERY=${ENERGY_EVERY:-$NSTEPS}
REPEATS=${REPEATS:-3}
THREADS=${THREADS:-"1 2 4 8"}
KERNELS=${KERNELS:-"direct direct-split2 direct-split4 direct-split8"}
INV_SQRT=${INV_SQRT:-libm}
OUTPUT_DIR=${OUTPUT_DIR:-results}
INPUT=${INPUT:-$OUTPUT_DIR/plummer_accumulator_${N}.bin}
CSV=${CSV:-$OUTPUT_DIR/accumulator_tradeoff_$(date +%Y%m%d_%H%M%S).csv}

mkdir -p "$OUTPUT_DIR"

make OPENMP=1 PRECISION=double

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
  "kernel,inv_sqrt,n,nsteps,dt,eps,mass,threads,repeat,total_seconds,force_seconds,energy_seconds,max_relative_energy_drift,status" \
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

for threads in $THREADS; do
  for kernel in $KERNELS; do
    for repeat in $(seq 1 "$REPEATS"); do
      echo "# accumulator tradeoff kernel=$kernel threads=$threads repeat=$repeat/$REPEATS"

      output=$(
        OMP_NUM_THREADS="$threads" \
        OMP_PROC_BIND="${OMP_PROC_BIND:-close}" \
        OMP_PLACES="${OMP_PLACES:-cores}" \
        ./nbody_direct_serial \
          --input "$INPUT" \
          --nsteps "$NSTEPS" \
          --dt "$DT" \
          --eps "$EPS" \
          --mass "$MASS" \
          --energy-every "$ENERGY_EVERY" \
          --force-kernel "$kernel" \
          --inv-sqrt "$INV_SQRT" \
          --timing \
          --quiet
      )

      printf '%s\n' "$output"

      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$kernel" \
        "$INV_SQRT" \
        "$N" \
        "$NSTEPS" \
        "$DT" \
        "$EPS" \
        "$MASS" \
        "$threads" \
        "$repeat" \
        "$(printf '%s\n' "$output" | extract_timing total_seconds)" \
        "$(printf '%s\n' "$output" | extract_timing force_seconds)" \
        "$(printf '%s\n' "$output" | extract_timing energy_seconds)" \
        "$(printf '%s\n' "$output" | extract_final_field max_relative_energy_drift)" \
        "$(printf '%s\n' "$output" | extract_final_field status)" \
        >> "$CSV"
    done
  done
done

echo "# wrote $CSV"
