#!/usr/bin/env bash
# Environment setup script for Ascend C ubench on 910B (CANN 9.0.0)
# Source this file to set up the build environment:
#   source setup_env_910b.sh

set -euo pipefail

# ── CANN toolkit path ────────────────────────────────────────────────────────
export ASCEND_CANN_PACKAGE_PATH="/usr/local/Ascend/cann-9.0.0"
export ASCEND_SOC_VERSION="ascend910_9362"
export ASCEND_OPT_LEVEL="O2"

# Source CANN environment
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [[ -f "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh" ]]; then
  source "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh"
fi

# Add ccec compiler to PATH (needed for cmake build)
export PATH="${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ccec_compiler/bin:${PATH}"

# ── Build paths ──────────────────────────────────────────────────────────────
export UBENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UBENCH_BUILD_DIR="${UBENCH_ROOT}/build_full"
export UBENCH_BIN_DIR="${UBENCH_BUILD_DIR}/bin"

# ── Verify environment ──────────────────────────────────────────────────────
echo "=== Ascend C ubench environment (910B) ==="
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
echo "To build:  bash scripts/build_one_click.sh"
echo "To run:    bash scripts/run_all.sh"
