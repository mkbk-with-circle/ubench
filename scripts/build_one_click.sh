#!/usr/bin/env bash
# One-click build script for Ascend C ubench
# This script configures the environment and builds all benchmarks.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Ascend C ubench one-click build ==="
echo "Root: ${ROOT_DIR}"

# ── Step 1: Source CANN environment ──────────────────────────────────────────
echo ""
echo "[1/3] Setting up CANN environment..."
export ASCEND_CANN_PACKAGE_PATH="/usr/local/Ascend/ascend-toolkit/8.2.RC1/aarch64-linux"
export ASCEND_SOC_VERSION="Ascend310P3"
export ASCEND_OPT_LEVEL="O2"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# ── Step 2: Fix kernel source issues ────────────────────────────────────────
echo "[2/3] Fixing kernel source compatibility issues..."

# Fix unnamed parameters (required by host stub generator)
for f in "${ROOT_DIR}"/mte/*/kernel.cpp "${ROOT_DIR}"/vector/*/kernel.cpp "${ROOT_DIR}"/cube/*.cpp "${ROOT_DIR}"/scalar/*.cpp; do
  if [[ -f "$f" ]]; then
    # Fix "uint32_t) {" -> "uint32_t mode) {"
    sed -i 's/uint32_t) {/uint32_t mode) {/g' "$f"
    # Fix "GM_ADDR," -> "GM_ADDR input," (for unnamed GM_ADDR parameters)
    sed -i 's/GM_ADDR,/GM_ADDR input,/g' "$f"
    # Fix "uint32_t, uint32_t repeats" -> "uint32_t size_bytes, uint32_t repeats"
    sed -i 's/uint32_t, uint32_t repeats/uint32_t size_bytes, uint32_t repeats/g' "$f"
  fi
done

# Fix static_cast<float>(uint) for DaVinci AI Core compatibility
for f in "${ROOT_DIR}"/vector/*/kernel.cpp "${ROOT_DIR}"/cube/*.cpp; do
  if [[ -f "$f" ]]; then
    sed -i 's/static_cast<float>(\(i\))/{ union { uint32_t u; float f; } c; c.u = \1; c.f }/g' "$f"
    sed -i 's/static_cast<float>(\(i + 1\))/{ union { uint32_t u; float f; } c; c.u = \1; c.f }/g' "$f"
  fi
done

# ── Step 3: Build ───────────────────────────────────────────────────────────
echo "[3/3] Building benchmarks..."
cd "${ROOT_DIR}"

# Check if local cmake patches exist
if [[ ! -d "${ROOT_DIR}/local_ascendc_cmake" ]]; then
  echo "Creating local cmake patches..."
  # Create local ascendc_cmake directory with ACL include path fix
  cp -r "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/ascendc_kernel_cmake" "${ROOT_DIR}/local_ascendc_cmake"
  chmod -R u+w "${ROOT_DIR}/local_ascendc_cmake"

  # Add ACL include path to precompile project
  cat >> "${ROOT_DIR}/local_ascendc_cmake/device_precompile_project/CMakeLists.txt" << 'PATCH'

# Additional include paths for ACL headers
target_include_directories(intf_device INTERFACE
    ${ASCEND_CANN_PACKAGE_PATH}/include
)
PATCH

  # Create local ascendc.cmake
  cat > "${ROOT_DIR}/local_ascendc_cmake/ascendc.cmake" << 'PATCH'
get_filename_component(ASCENDC_KERNEL_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
include(${ASCENDC_KERNEL_CMAKE_DIR}/host_config.cmake)
include(${ASCENDC_KERNEL_CMAKE_DIR}/host_intf.cmake)
include(${ASCENDC_KERNEL_CMAKE_DIR}/function.cmake)
PATCH
fi

# Create local include directory with ACL headers
mkdir -p "${ROOT_DIR}/local_include/acl"
if [[ ! -L "${ROOT_DIR}/local_include/acl/acl.h" ]]; then
  ln -sf "${ASCEND_CANN_PACKAGE_PATH}/include/acl/"* "${ROOT_DIR}/local_include/acl/"
fi

# Create cann_wrapper directory
if [[ ! -d "${ROOT_DIR}/cann_wrapper" ]]; then
  echo "Creating CANN wrapper directory..."
  mkdir -p "${ROOT_DIR}/cann_wrapper"
  cd "${ROOT_DIR}/cann_wrapper"
  ln -sf "${ASCEND_CANN_PACKAGE_PATH}"/* .
  rm -f ascendc_devkit
  mkdir -p ascendc_devkit
  ln -sf "${ASCEND_CANN_PACKAGE_PATH}/ccec_compiler" ascendc_devkit/ccec_compiler
  ln -sf "${ASCEND_CANN_PACKAGE_PATH}/tikcpp" ascendc_devkit/tikcpp
  ln -sf "${ASCEND_CANN_PACKAGE_PATH}/include" ascendc_devkit/include
  ln -sf "${ASCEND_CANN_PACKAGE_PATH}/lib64" ascendc_devkit/lib64
  ln -sf "${ASCEND_CANN_PACKAGE_PATH}/bin" ascendc_devkit/bin
  cd "${ROOT_DIR}"
fi

# Create device_includes directory with minimal acl stub
mkdir -p "${ROOT_DIR}/device_includes/acl"
cat > "${ROOT_DIR}/device_includes/acl/acl.h" << 'STUB'
#ifndef ACL_ACL_H_DEVICE_COMPAT
#define ACL_ACL_H_DEVICE_COMPAT
#if defined(__CCE__) && !defined(ASCENDC_CPU_DEBUG)
#include <cstdint>
typedef int aclError;
typedef void* aclrtContext;
typedef void* aclrtStream;
#else
#include_next <acl/acl.h>
#endif
#endif
STUB

# Build
rm -rf build_full
mkdir build_full
cmake -S . -B build_full \
  -DASCEND_CANN_PACKAGE_PATH="${ROOT_DIR}/cann_wrapper" \
  -DASCEND_SOC_VERSION="${ASCEND_SOC_VERSION}" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build_full -j"$(nproc)"

echo ""
echo "=== Build complete ==="
echo "Binaries in ${ROOT_DIR}/build_full/bin/"
ls -la "${ROOT_DIR}/build_full/bin/"
