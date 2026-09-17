#!/bin/sh
set -eu

N=${N:-16777216}
REPEATS=${REPEATS:-5}
OUTPUT_DIR=${OUTPUT_DIR:-results}
CSV=${CSV:-$OUTPUT_DIR/rsqrt_kernel_$(date +%Y%m%d_%H%M%S).csv}
CFLAGS=${CFLAGS:-"-O3 -march=native -ffp-contract=fast -Wall -Wextra -Wpedantic"}

mkdir -p "$OUTPUT_DIR"

make PRECISION=float benchmark_rsqrt_kernel CFLAGS="$CFLAGS"

echo "# rsqrt kernel microbenchmark"
echo "# n=$N repeats=$REPEATS"
echo "# csv=$CSV"
echo "# cflags=$CFLAGS"

./benchmark_rsqrt_kernel \
  --n "$N" \
  --repeats "$REPEATS" \
  > "$CSV"

echo "# wrote $CSV"
