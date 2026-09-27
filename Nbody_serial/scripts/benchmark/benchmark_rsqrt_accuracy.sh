#!/bin/sh
set -eu

N=${N:-8192}
EPS=${EPS:-0.05}
MASS=${MASS:-1.0}
SEED=${SEED:-123}
OUTPUT_DIR=${OUTPUT_DIR:-results}
REPORT_DIR=${REPORT_DIR:-report/tables}
INPUT=${INPUT:-$OUTPUT_DIR/plummer_rsqrt_${N}.bin}
CSV=${CSV:-$REPORT_DIR/rsqrt_accuracy_summary.csv}
MARKDOWN=${MARKDOWN:-$REPORT_DIR/rsqrt_accuracy_summary.md}

mkdir -p "$OUTPUT_DIR" "$REPORT_DIR"

make PRECISION=double generate_ic benchmark_rsqrt_accuracy

if [ ! -f "$INPUT" ]; then
  ./generate_ic \
    --model 0 \
    --n "$N" \
    --seed "$SEED" \
    --scale 1.0 \
    --mass "$MASS" \
    --output "$INPUT"
fi

./benchmark_rsqrt_accuracy \
  --input "$INPUT" \
  --eps "$EPS" \
  --mass "$MASS" \
  > "$CSV"

awk -F, '
  BEGIN {
    print "| Variant | Max relative accel error | RMS relative accel error | Max absolute accel error | Reference accel RMS | Status |"
    print "| :------ | -----------------------: | -----------------------: | -----------------------: | ------------------: | :----- |"
  }
  NR > 1 {
    printf "| %s | %.6e | %.6e | %.6e | %.6e | %s |\n", $1, $7, $8, $9, $10, $11
  }
' "$CSV" > "$MARKDOWN"

cat "$MARKDOWN"
echo "# wrote $CSV"
echo "# wrote $MARKDOWN"
