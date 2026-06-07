#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${ROOT_DIR}/build_full/bin"
RESULT_DIR="${ROOT_DIR}/results"
mkdir -p "${RESULT_DIR}"

COMMON_ARGS=(
  --device "${DEVICE_ID:-0}"
  --warmup "${WARMUP:-5}"
  --iters "${ITERS:-20}"
  --blocks "${BLOCKS:-8}"
  --repeats "${REPEATS:-1000}"
  --size "${SIZE_BYTES:-1048576}"
)

BENCHES=(
  mte_copy_bw
  mte_startup_latency
  mte_granularity
  vector_add_latency
  vector_mul_latency
  vector_throughput
  vector_pipeline_depth
  cube_tile_latency
  cube_throughput
  cube_scaling
  scalar_arith_latency
  scalar_branch_overhead
)

for bench in "${BENCHES[@]}"; do
  exe="${BIN_DIR}/${bench}"
  if [[ ! -x "${exe}" ]]; then
    echo "[skip] ${bench}: executable not found at ${exe}" >&2
    continue
  fi
  echo "===== ${bench} ====="
  "${exe}" "${COMMON_ARGS[@]}" --csv "${RESULT_DIR}/${bench}.csv" \
    | tee "${RESULT_DIR}/${bench}.txt"
done

"${ROOT_DIR}/scripts/summarize_csv.sh" "${RESULT_DIR}"/*.csv \
  > "${RESULT_DIR}/summary.csv"

echo "Summary written to ${RESULT_DIR}/summary.csv"
