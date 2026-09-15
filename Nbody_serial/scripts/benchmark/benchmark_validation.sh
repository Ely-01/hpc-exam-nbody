#!/bin/sh
set -eu

N=${N:-10000}
NSTEPS=${NSTEPS:-100}
DT=${DT:-1e-4}
EPS=${EPS:-0.05}
MASS=${MASS:-1.0}
SEED=${SEED:-123}
ENERGY_EVERY=${ENERGY_EVERY:-10}
THREADS=${THREADS:-${OMP_NUM_THREADS:-1}}
INTEGRATOR=${INTEGRATOR:-kdk}
FORCE_KERNEL=${FORCE_KERNEL:-direct}
INV_SQRT=${INV_SQRT:-libm}
OUTPUT_DIR=${OUTPUT_DIR:-results}
REPORT_DIR=${REPORT_DIR:-report/tables}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
INPUT=${INPUT:-$OUTPUT_DIR/plummer_${N}_validation.bin}
LOG=${LOG:-$OUTPUT_DIR/validation_energy_${TIMESTAMP}.log}
CSV=${CSV:-$REPORT_DIR/validation_energy_summary.csv}
MARKDOWN=${MARKDOWN:-$REPORT_DIR/validation_energy_summary.md}
CFLAGS=${CFLAGS:-"-O3 -march=native -ffp-contract=fast -Wall -Wextra -Wpedantic"}

mkdir -p "$OUTPUT_DIR" "$REPORT_DIR"

make OPENMP=1 PRECISION=double CFLAGS="$CFLAGS"

if [ ! -f "$INPUT" ] || [ "${FORCE_REGENERATE:-0}" = "1" ]; then
  ./generate_ic \
    --model 0 \
    --n "$N" \
    --seed "$SEED" \
    --scale 1.0 \
    --mass "$MASS" \
    --output "$INPUT"
fi

echo "# validation run"
echo "# input=$INPUT"
echo "# n=$N nsteps=$NSTEPS dt=$DT eps=$EPS mass=$MASS seed=$SEED"
echo "# threads=$THREADS integrator=$INTEGRATOR force_kernel=$FORCE_KERNEL inv_sqrt=$INV_SQRT"
echo "# log=$LOG"

OMP_NUM_THREADS="$THREADS" ./nbody_direct_serial \
  --input "$INPUT" \
  --nsteps "$NSTEPS" \
  --dt "$DT" \
  --eps "$EPS" \
  --mass "$MASS" \
  --energy-every "$ENERGY_EVERY" \
  --integrator "$INTEGRATOR" \
  --force-kernel "$FORCE_KERNEL" \
  --inv-sqrt "$INV_SQRT" \
  --timing \
  --quiet \
  > "$LOG"

cat "$LOG"

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
  ' "$LOG"
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
  ' "$LOG"
}

drift=$(extract_final_field max_relative_energy_drift)
status=$(extract_final_field status)
total_seconds=$(extract_timing total_seconds)
force_seconds=$(extract_timing force_seconds)
energy_seconds=$(extract_timing energy_seconds)

cat > "$CSV" <<EOF
n,nsteps,dt,eps,mass,seed,threads,integrator,force_kernel,inv_sqrt,total_seconds,force_seconds,energy_seconds,max_relative_energy_drift,status,log
$N,$NSTEPS,$DT,$EPS,$MASS,$SEED,$THREADS,$INTEGRATOR,$FORCE_KERNEL,$INV_SQRT,$total_seconds,$force_seconds,$energy_seconds,$drift,$status,$LOG
EOF

cat > "$MARKDOWN" <<EOF
| N | Steps | dt | eps | Threads | Integrator | Force kernel | inv sqrt | Total s | Force s | Energy s | Max drift | Status |
|---:|---:|---:|---:|---:|:---|:---|:---|---:|---:|---:|---:|:---|
| $N | $NSTEPS | $DT | $EPS | $THREADS | $INTEGRATOR | $FORCE_KERNEL | $INV_SQRT | $total_seconds | $force_seconds | $energy_seconds | $drift | $status |
EOF

echo "# wrote $CSV"
echo "# wrote $MARKDOWN"
