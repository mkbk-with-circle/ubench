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

# ── Cube 跳过逻辑 ──────────────────────────────────────────────────────────
# SKIP_CUBE=1 显式跳过；或自动检测 310P 平台
SKIP_CUBE="${SKIP_CUBE:-}"

if [[ -z "${SKIP_CUBE}" ]]; then
  # 自动检测: npu-smi 中出现 "310P" 则跳过 Cube
  if command -v npu-smi &>/dev/null; then
    if npu-smi info 2>/dev/null | grep -q "310P"; then
      SKIP_CUBE=1
      echo "[auto-detect] Ascend 310P detected, skipping Cube benchmarks"
    fi
  fi
fi

CUBE_BENCHES=(cube_tile_latency cube_throughput cube_scaling)

# ── 基准列表 ──────────────────────────────────────────────────────────────
BENCHES=(
  mte_copy_bw
  mte_startup_latency
  mte_granularity
  vector_add_latency
  vector_mul_latency
  vector_throughput
  vector_pipeline_depth
  scalar_arith_latency
  scalar_branch_overhead
)

# 根据 SKIP_CUBE 决定是否加入 Cube
if [[ "${SKIP_CUBE}" != "1" ]]; then
  BENCHES+=("${CUBE_BENCHES[@]}")
fi

# ── 运行基准 ──────────────────────────────────────────────────────────────
for bench in "${BENCHES[@]}"; do
  exe="${BIN_DIR}/${bench}"
  if [[ ! -x "${exe}" ]]; then
    echo "[skip] ${bench}: executable not found at ${exe}" >&2
    continue
  fi
  echo "===== ${bench} ====="
  timeout "${BENCH_TIMEOUT:-120}" \
    "${exe}" "${COMMON_ARGS[@]}" --csv "${RESULT_DIR}/${bench}.csv" \
    | tee "${RESULT_DIR}/${bench}.txt"
done

# ── Cube 跳过处理 ──────────────────────────────────────────────────────────
if [[ "${SKIP_CUBE}" == "1" ]]; then
  SKIP_FILE="${RESULT_DIR}/cube_skipped_310p.txt"
  cat > "${SKIP_FILE}" << 'EOF'
Cube benchmarks skipped on Ascend 310P.

Reason: Matmul template (lib/matmul_intf.h) hangs on 310P3 (m200).
  - IterateAll() v200 path calls DataCacheCleanAndInvalid on address 0
  - Scheduler fails to initialize even with correct TCubeTiling
  - MmadImpl() also hangs — issue at Cube unit driver level

Skipped benchmarks:
  - cube_tile_latency
  - cube_throughput
  - cube_scaling

These need to be re-run on 910B hardware.
EOF
  echo "[skip] Cube benchmarks skipped (310P), reason written to ${SKIP_FILE}"
  # 写空 .txt 避免 summarize 报错
  for cb in "${CUBE_BENCHES[@]}"; do
    : > "${RESULT_DIR}/${cb}.txt"
    # 不写 .csv，让 summarize 跳过
  done
fi

# ── 汇总 ──────────────────────────────────────────────────────────────────
CSV_FILES=("${RESULT_DIR}"/*.csv)
if [[ ${#CSV_FILES[@]} -gt 0 && -f "${CSV_FILES[0]}" ]]; then
  "${ROOT_DIR}/scripts/summarize_csv.sh" "${CSV_FILES[@]}" \
    > "${RESULT_DIR}/summary.csv"
  echo "Summary written to ${RESULT_DIR}/summary.csv"
else
  echo "[warn] No CSV files found, skipping summary" >&2
fi
