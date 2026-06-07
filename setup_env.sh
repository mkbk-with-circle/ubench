#!/usr/bin/env bash
# Environment setup script for Ascend C ubench
# Source this file to set up the build environment:
#   source setup_env.sh

set -euo pipefail

# ── CANN toolkit path ────────────────────────────────────────────────────────
export ASCEND_CANN_PACKAGE_PATH="/usr/local/Ascend/ascend-toolkit/8.2.RC1/aarch64-linux"
export ASCEND_SOC_VERSION="Ascend310P3"
export ASCEND_OPT_LEVEL="O2"

# Source CANN environment
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# ── Build paths ──────────────────────────────────────────────────────────────
export UBENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UBENCH_BUILD_DIR="${UBENCH_ROOT}/build_full"
export UBENCH_BIN_DIR="${UBENCH_BUILD_DIR}/bin"

# ── Verify environment ──────────────────────────────────────────────────────
echo "=== Ascend C ubench environment ==="
echo "CANN path:     ${ASCEND_CANN_PACKAGE_PATH}"
echo "SoC version:   ${ASCEND_SOC_VERSION}"
echo "Opt level:     ${ASCEND_OPT_LEVEL}"
echo "Build dir:     ${UBENCH_BUILD_DIR}"
echo "Bin dir:       ${UBENCH_BIN_DIR}"
echo ""

# Check NPU availability
if command -v npu-smi &>/dev/null; then
  echo "NPU devices:"
  npu-smi info 2>/dev/null | grep -E "^\| [0-9]" | head -8
else
  echo "Warning: npu-smi not found"
fi

echo ""
echo "To build:  cmake --build ${UBENCH_BUILD_DIR} -j\$(nproc)"
echo "To run:    bash scripts/run_all.sh"
