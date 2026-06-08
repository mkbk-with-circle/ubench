#!/usr/bin/env bash
# One-click build and run script for Ascend 910B (CANN 9.0.0)
# Usage: bash scripts/build_and_run_910b.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Ascend C ubench: 910B build and run ==="
echo "Root: ${ROOT_DIR}"

# ── Environment setup ────────────────────────────────────────────────────────
export ASCEND_CANN_PACKAGE_PATH="/usr/local/Ascend/cann-9.0.0"
export ASCEND_SOC_VERSION="ascend910_9362"
export ASCEND_OPT_LEVEL="O2"

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export PATH="${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ccec_compiler/bin:${PATH}"

# ── Build ────────────────────────────────────────────────────────────────────
echo ""
echo "=== Building ==="
cd "${ROOT_DIR}"
rm -rf build_full
mkdir build_full

cmake -S . -B build_full \
  -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH}" \
  -DASCEND_SOC_VERSION="${ASCEND_SOC_VERSION}" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build_full -j"$(nproc)"

echo ""
echo "=== Build complete ==="
ls -la "${ROOT_DIR}/build_full/bin/"

# ── Run ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== Running benchmarks ==="
export DEVICE_ID=0
export WARMUP=5
export ITERS=20
export BLOCKS=4
export REPEATS=1000
export SIZE_BYTES=65536

bash "${ROOT_DIR}/scripts/run_all.sh"

echo ""
echo "=== Results ==="
echo "Results in: ${ROOT_DIR}/results/"
cat "${ROOT_DIR}/results/summary.csv"
