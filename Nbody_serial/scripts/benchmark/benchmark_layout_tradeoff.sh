#!/bin/sh
set -eu

N=${N:-8192}
REPEATS=${REPEATS:-5}
THREADS=${THREADS:-"1 2 4 8"}
EPS=${EPS:-0.05}
MASS=${MASS:-1.0}
SEED=${SEED:-123}
OUTPUT_DIR=${OUTPUT_DIR:-results}
INPUT=${INPUT:-$OUTPUT_DIR/plummer_layout_${N}.bin}
CSV=${CSV:-$OUTPUT_DIR/layout_tradeoff_final.csv}
CFLAGS=${CFLAGS:-"-O3 -march=native -ffp-contract=fast -Wall -Wextra -Wpedantic"}

mkdir -p "$OUTPUT_DIR"

make OPENMP=1 PRECISION=double benchmark_layout CFLAGS="$CFLAGS"
make PRECISION=double generate_ic CFLAGS="$CFLAGS"

if [ ! -f "$INPUT" ] || [ "${FORCE_REGENERATE:-0}" = "1" ]; then
  ./generate_ic \
    --model 0 \
    --n "$N" \
    --seed "$SEED" \
    --scale 1.0 \
    --mass "$MASS" \
    --output "$INPUT"
fi

echo "# layout benchmark"
echo "# input=$INPUT"
echo "# n=$N repeats=$REPEATS threads=$THREADS eps=$EPS mass=$MASS"
echo "# csv=$CSV"

first=1
for threads in $THREADS; do
  if [ "$first" = "1" ]; then
    OMP_NUM_THREADS="$threads" ./benchmark_layout \
      --input "$INPUT" \
      --repeats "$REPEATS" \
      --eps "$EPS" \
      --mass "$MASS" \
      > "$CSV"
    first=0
  else
    OMP_NUM_THREADS="$threads" ./benchmark_layout \
      --input "$INPUT" \
      --repeats "$REPEATS" \
      --eps "$EPS" \
      --mass "$MASS" \
      --no-header \
      >> "$CSV"
  fi
done

echo "# wrote $CSV"
