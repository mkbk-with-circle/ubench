#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${ASCEND_CANN_PACKAGE_PATH:=/usr/local/Ascend/ascend-toolkit/latest}"
: "${ASCEND_SOC_VERSION:=Ascend310P3}"
: "${ASCEND_OPT_LEVEL:=O2}"

cat <<EOF
Ascend C device-kernel build hook
=================================

This repository keeps host runners and device kernels separate because CANN
Kernel Launch projects differ across CANN releases.

Use the CANN Kernel Launch project flow for ${ASCEND_SOC_VERSION}, then link the
generated object/static library so it overrides common/launch_stubs.cpp weak
symbols.

Kernel sources:
  ${ROOT_DIR}/mte/copy_bw/kernel.cpp
  ${ROOT_DIR}/mte/startup_latency/kernel.cpp
  ${ROOT_DIR}/mte/granularity/kernel.cpp
  ${ROOT_DIR}/vector/add_latency/kernel.cpp
  ${ROOT_DIR}/vector/mul_latency/kernel.cpp
  ${ROOT_DIR}/vector/throughput/kernel.cpp
  ${ROOT_DIR}/vector/pipeline_depth/kernel.cpp
  ${ROOT_DIR}/cube/cube_matmul_kernel.cpp
  ${ROOT_DIR}/scalar/scalar_kernel.cpp

Required build metadata:
  ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
  ASCEND_SOC_VERSION=${ASCEND_SOC_VERSION}
  ASCEND_OPT_LEVEL=-${ASCEND_OPT_LEVEL}

The weak stubs intentionally fail at runtime if the actual Ascend C launch
symbols are not linked. This prevents accidentally trusting host-only builds.
EOF
