#!/bin/sh
set -eu

VALIDATION_MODE=${VALIDATION_MODE:-energy}
N=${N:-10000}
N_BASE=${N_BASE:-1000}
N_FACTOR=${N_FACTOR:-10}
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
CFLAGS=${CFLAGS:-"-O3 -march=native -ffp-contract=fast -Wall -Wextra -Wpedantic"}

mkdir -p "$OUTPUT_DIR" "$REPORT_DIR"

make OPENMP=1 PRECISION=double CFLAGS="$CFLAGS"

extract_final_field () {
  log=$1
  key=$2
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
  ' "$log"
}

extract_timing () {
  log=$1
  key=$2
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
  ' "$log"
}

run_case () {
  n=$1
  label=$2
  input=${INPUT:-$OUTPUT_DIR/plummer_${n}_${label}.bin}
  log=$OUTPUT_DIR/validation_${label}_${TIMESTAMP}.log

  if [ ! -f "$input" ] || [ "${FORCE_REGENERATE:-0}" = "1" ]; then
    ./generate_ic \
      --model 0 \
      --n "$n" \
      --seed "$SEED" \
      --scale 1.0 \
      --mass "$MASS" \
      --output "$input"
  fi

  echo "# validation run" >&2
  echo "# mode=$VALIDATION_MODE input=$input" >&2
  echo "# n=$n nsteps=$NSTEPS dt=$DT eps=$EPS mass=$MASS seed=$SEED" >&2
  echo "# threads=$THREADS integrator=$INTEGRATOR force_kernel=$FORCE_KERNEL inv_sqrt=$INV_SQRT" >&2
  echo "# log=$log" >&2

  OMP_NUM_THREADS="$THREADS" ./nbody_direct_serial \
    --input "$input" \
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
    > "$log"

  cat "$log" >&2
  printf '%s\n' "$log"
}

write_energy_summary () {
  log=$1
  csv=${CSV:-$REPORT_DIR/validation_energy_summary.csv}
  markdown=${MARKDOWN:-$REPORT_DIR/validation_energy_summary.md}

  drift=$(extract_final_field "$log" max_relative_energy_drift)
  status=$(extract_final_field "$log" status)
  total_seconds=$(extract_timing "$log" total_seconds)
  force_seconds=$(extract_timing "$log" force_seconds)
  energy_seconds=$(extract_timing "$log" energy_seconds)

  cat > "$csv" <<EOF
n,nsteps,dt,eps,mass,seed,threads,integrator,force_kernel,inv_sqrt,total_seconds,force_seconds,energy_seconds,max_relative_energy_drift,status,log
$N,$NSTEPS,$DT,$EPS,$MASS,$SEED,$THREADS,$INTEGRATOR,$FORCE_KERNEL,$INV_SQRT,$total_seconds,$force_seconds,$energy_seconds,$drift,$status,$log
EOF

  cat > "$markdown" <<EOF
| N | Steps | dt | eps | Threads | Integrator | Force kernel | inv sqrt | Total s | Force s | Energy s | Max drift | Status |
|---:|---:|---:|---:|---:|:---|:---|:---|---:|---:|---:|---:|:---|
| $N | $NSTEPS | $DT | $EPS | $THREADS | $INTEGRATOR | $FORCE_KERNEL | $INV_SQRT | $total_seconds | $force_seconds | $energy_seconds | $drift | $status |
EOF

  echo "# wrote $csv"
  echo "# wrote $markdown"
}

write_growth_summary () {
  small_log=$1
  large_log=$2
  n_large=$3
  csv=${CSV:-$REPORT_DIR/n_growth_summary.csv}
  markdown=${MARKDOWN:-$REPORT_DIR/n_growth_summary.md}

  small_force=$(extract_timing "$small_log" force_seconds)
  large_force=$(extract_timing "$large_log" force_seconds)
  small_total=$(extract_timing "$small_log" total_seconds)
  large_total=$(extract_timing "$large_log" total_seconds)
  small_drift=$(extract_final_field "$small_log" max_relative_energy_drift)
  large_drift=$(extract_final_field "$large_log" max_relative_energy_drift)
  small_status=$(extract_final_field "$small_log" status)
  large_status=$(extract_final_field "$large_log" status)

  awk -v n0="$N_BASE" -v n1="$n_large" \
      -v f0="$small_force" -v f1="$large_force" \
      -v steps="$NSTEPS" -v total0="$small_total" -v total1="$large_total" \
      -v drift0="$small_drift" -v drift1="$large_drift" \
      -v status0="$small_status" -v status1="$large_status" \
      -v threads="$THREADS" -v dt="$DT" -v eps="$EPS" \
      -v csv="$csv" -v markdown="$markdown" '
    BEGIN {
      force_step0 = f0 / steps
      force_step1 = f1 / steps
      measured_ratio = force_step1 / force_step0
      expected_ratio = (n1 * (n1 - 1)) / (n0 * (n0 - 1))

      print "n,nsteps,dt,eps,threads,total_seconds,force_seconds,force_seconds_per_step,max_relative_energy_drift,status" > csv
      printf "%d,%d,%s,%s,%d,%.9f,%.9f,%.12f,%s,%s\n", n0, steps, dt, eps, threads, total0, f0, force_step0, drift0, status0 >> csv
      printf "%d,%d,%s,%s,%d,%.9f,%.9f,%.12f,%s,%s\n", n1, steps, dt, eps, threads, total1, f1, force_step1, drift1, status1 >> csv
      printf "# expected_pair_ratio,%.6f\n# measured_force_per_step_ratio,%.6f\n", expected_ratio, measured_ratio >> csv

      print "| N | Steps | dt | eps | Threads | Total s | Force s | Force s/step | Max drift | Status |" > markdown
      print "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|" >> markdown
      printf "| %d | %d | %s | %s | %d | %.6f | %.6f | %.9f | %s | %s |\n", n0, steps, dt, eps, threads, total0, f0, force_step0, drift0, status0 >> markdown
      printf "| %d | %d | %s | %s | %d | %.6f | %.6f | %.9f | %s | %s |\n", n1, steps, dt, eps, threads, total1, f1, force_step1, drift1, status1 >> markdown
      print "" >> markdown
      printf "Expected pair-count ratio: %.3f\n\n", expected_ratio >> markdown
      printf "Measured force-time-per-step ratio: %.3f\n", measured_ratio >> markdown
    }
  '

  echo "# wrote $csv"
  echo "# wrote $markdown"
}

case "$VALIDATION_MODE" in
  energy)
    LOG=$(run_case "$N" "energy")
    write_energy_summary "$LOG"
    ;;
  growth)
    N_LARGE=$((N_BASE * N_FACTOR))
    SMALL_LOG=$(run_case "$N_BASE" "growth_${N_BASE}")
    LARGE_LOG=$(run_case "$N_LARGE" "growth_${N_LARGE}")
    write_growth_summary "$SMALL_LOG" "$LARGE_LOG" "$N_LARGE"
    ;;
  *)
    echo "error: VALIDATION_MODE must be energy or growth" >&2
    exit 1
    ;;
esac
