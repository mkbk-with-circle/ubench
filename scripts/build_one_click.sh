#!/usr/bin/env bash
# One-click build script for Ascend C ubench
# This script configures the environment and builds all benchmarks.
# Target: Ascend 910B with CANN 9.0.0
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Ascend C ubench one-click build ==="
echo "Root: ${ROOT_DIR}"

# ── Step 1: Source CANN environment ──────────────────────────────────────────
echo ""
echo "[1/2] Setting up CANN environment..."
export ASCEND_CANN_PACKAGE_PATH="/usr/local/Ascend/cann-9.0.0"
export ASCEND_SOC_VERSION="ascend910b"
export ASCEND_OPT_LEVEL="O2"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [[ -f "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh" ]]; then
  source "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh"
fi

# ── Step 2: Build ───────────────────────────────────────────────────────────
echo "[2/2] Building benchmarks..."
cd "${ROOT_DIR}"

# Clean previous build
rm -rf build_full
mkdir build_full

# Use CANN 9.0.0's built-in cmake modules (no patches needed)
cmake -S . -B build_full \
  -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH}" \
  -DASCEND_SOC_VERSION="${ASCEND_SOC_VERSION}" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build_full -j"$(nproc)"

echo ""
echo "=== Build complete ==="
echo "Binaries in ${ROOT_DIR}/build_full/bin/"
ls -la "${ROOT_DIR}/build_full/bin/"
