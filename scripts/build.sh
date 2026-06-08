#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [[ -n "${ASCEND_CANN_PACKAGE_PATH:-}" && -f "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh"
fi

: "${ASCEND_CANN_PACKAGE_PATH:=/usr/local/Ascend/cann-9.0.0}"
: "${ASCEND_SOC_VERSION:=ascend910b}"
: "${ASCEND_OPT_LEVEL:=O2}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH}" \
  -DASCEND_SOC_VERSION="${ASCEND_SOC_VERSION}" \
  -DASCEND_OPT_LEVEL="${ASCEND_OPT_LEVEL}"

cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "Host binaries are in ${BUILD_DIR}/bin"
echo "Build/link Ascend C device kernels with scripts/build_kernels.sh before running."
