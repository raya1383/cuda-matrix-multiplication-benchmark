#!/usr/bin/env bash
set -euo pipefail

BINARY="${BINARY:-./build/bin/matrix_bench}"
CSV="${CSV:-results/sparsity_sweep.csv}"
N="${N:-2048}"
ITERS="${ITERS:-7}"
WARMUP="${WARMUP:-2}"

mkdir -p "$(dirname "$CSV")"
rm -f "$CSV"

# zero probability is swept to show when cuSPARSE becomes useful.
for dtype in float double; do
  for zero_prob in 0.0 0.50 0.90 0.95 0.99; do
    echo "N=$N type=$dtype zero_prob=$zero_prob"
    "$BINARY" "$N" "$dtype" \
      --methods cuda_tiled,cublas,cusparse \
      --iters "$ITERS" --warmup "$WARMUP" \
      --zero-prob "$zero_prob" --verify 1 --csv "$CSV"
  done
done

echo "Sparsity sweep saved to $CSV"
