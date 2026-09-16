#!/usr/bin/env bash
set -euo pipefail

# Edit these values for your GPU and available execution time.
BINARY="${BINARY:-./build/bin/matrix_bench}"
CSV="${CSV:-results/results.csv}"
ITERS="${ITERS:-7}"
WARMUP="${WARMUP:-2}"
CPU_THREADS="${CPU_THREADS:-$(nproc)}"
ZERO_PROB="${ZERO_PROB:-0.0}"
SIZES=(${SIZES:-64 128 256 512 768 1024 1536 2048})
TYPES=(${TYPES:-int float double})

mkdir -p "$(dirname "$CSV")"
rm -f "$CSV"

for dtype in "${TYPES[@]}"; do
  for n in "${SIZES[@]}"; do
    echo "============================================================"
    echo "Running N=$n type=$dtype"
    "$BINARY" "$n" "$dtype" \
      --iters "$ITERS" \
      --warmup "$WARMUP" \
      --cpu-threads "$CPU_THREADS" \
      --cpu-naive-max 768 \
      --cpu-blocked-max 2048 \
      --zero-prob "$ZERO_PROB" \
      --verify 1 \
      --csv "$CSV"
  done
done

echo "Results saved to $CSV"
