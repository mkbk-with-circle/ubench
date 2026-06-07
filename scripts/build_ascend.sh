#!/usr/bin/env bash
# Build script for Ascend C ubench using bisheng compiler
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_ascend"
BIN_DIR="${BUILD_DIR}/bin"
OBJ_DIR="${BUILD_DIR}/obj"

# CANN paths
CANN="${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/ascend-toolkit/8.2.RC1/aarch64-linux}"
BISHENG="${CANN}/ccec_compiler/bin/bisheng"
CCEC="${CANN}/ccec_compiler/bin/ccec"
PACK_KERNEL="${CANN}/bin/ascendc_pack_kernel"
TIKCPP="${CANN}/tikcpp/tikcfw"

mkdir -p "${BIN_DIR}" "${OBJ_DIR}"

echo "=== Building Ascend C ubench ==="
echo "CANN: ${CANN}"
echo "Bisheng: ${BISHENG}"

# ── Device kernel compilation ────────────────────────────────────────────────
DEVICE_FLAGS=(
  -I"${ROOT_DIR}/device_includes"
  -I"${TIKCPP}" -I"${TIKCPP}/interface" -I"${TIKCPP}/impl"
  -I"${CANN}/include"
  --cce-aicore-arch=dav-m200
  --cce-aicore-only --cce-auto-sync --cce-mask-opt
  --cce-disable-kernel-global-attr-check
  -mllvm -cce-aicore-fp-ceiling=2
  -mllvm -cce-aicore-record-overflow=false
  -mllvm -cce-aicore-mask-opt=false
  -O3 -std=c++17 --cce-aicore-lang
)

# ── Host kernel compilation (bisheng in host mode) ───────────────────────────
HOST_KERNEL_FLAGS=(
  -I"${TIKCPP}" -I"${TIKCPP}/interface" -I"${TIKCPP}/impl"
  -I"${CANN}/include"
  --cce-host-only --cce-aicore-lang
  -fPIC -O2 -std=c++17
)

# ── Host binary compilation (g++) ────────────────────────────────────────────
HOST_CXX="${CXX:-g++}"
HOST_FLAGS=(
  -O2 -std=c++17 -fPIC
  -I"${ROOT_DIR}/common"
  -I"${CANN}/include"
  -I"${CANN}/include/aclnn"
  -DASCEND_SOC_VERSION="Ascend310P3"
  -DASCEND_OPT_LEVEL="O2"
)

HOST_LINK_FLAGS=(
  -L"${CANN}/lib64"
  -L"${CANN}/runtime/lib64"
  -lascendcl -lpthread -ldl
)

# ── Kernel sources ───────────────────────────────────────────────────────────
KERNELS=(
  "mte/copy_bw/kernel.cpp:mte_copy_bw"
  "mte/startup_latency/kernel.cpp:mte_startup_latency"
  "mte/granularity/kernel.cpp:mte_granularity"
  "vector/add_latency/kernel.cpp:vector_add_latency"
  "vector/mul_latency/kernel.cpp:vector_mul_latency"
  "vector/throughput/kernel.cpp:vector_throughput"
  "vector/pipeline_depth/kernel.cpp:vector_pipeline_depth"
  "cube/cube_matmul_kernel.cpp:cube_matmul"
  "scalar/scalar_kernel.cpp:scalar"
)

# ── Benchmark definitions ────────────────────────────────────────────────────
BENCHES=(
  "mte_copy_bw:mte/copy_bw/main.cpp:mte_copy_bw"
  "mte_startup_latency:mte/startup_latency/main.cpp:mte_startup_latency"
  "mte_granularity:mte/granularity/main.cpp:mte_granularity"
  "vector_add_latency:vector/add_latency/main.cpp:vector_add_latency"
  "vector_mul_latency:vector/mul_latency/main.cpp:vector_mul_latency"
  "vector_throughput:vector/throughput/main.cpp:vector_throughput"
  "vector_pipeline_depth:vector/pipeline_depth/main.cpp:vector_pipeline_depth"
  "cube_tile_latency:cube/tile_latency/main.cpp:cube_matmul"
  "cube_throughput:cube/throughput/main.cpp:cube_matmul"
  "cube_scaling:cube/scaling/main.cpp:cube_matmul"
  "scalar_arith_latency:scalar/arith_latency/main.cpp:scalar"
  "scalar_branch_overhead:scalar/branch_overhead/main.cpp:scalar"
)

COMMON_SRCS=(
  "common/acl_utils.cpp"
  "common/bench_utils.cpp"
)

# ── Step 1: Compile device kernels ───────────────────────────────────────────
echo ""
echo "=== Step 1: Compile device kernels ==="
for entry in "${KERNELS[@]}"; do
  src="${ROOT_DIR}/${entry%%:*}"
  name="${entry##*:}"
  out="${OBJ_DIR}/${name}_device.o"
  echo "  ${name} (device)"
  "${CCEC}" "${DEVICE_FLAGS[@]}" -o "${out}" -c "${src}" 2>&1 | head -3
done

# ── Step 2: Compile host kernel objects ──────────────────────────────────────
echo ""
echo "=== Step 2: Compile host kernel objects ==="
for entry in "${KERNELS[@]}"; do
  src="${ROOT_DIR}/${entry%%:*}"
  name="${entry##*:}"
  out="${OBJ_DIR}/${name}_host.o"
  echo "  ${name} (host)"
  "${BISHENG}" "${HOST_KERNEL_FLAGS[@]}" -o "${out}" -c "${src}" 2>&1 | head -3
done

# ── Step 3: Compile common host objects ──────────────────────────────────────
echo ""
echo "=== Step 3: Compile common host objects ==="
for src in "${COMMON_SRCS[@]}"; do
  name=$(basename "${src}" .cpp)
  out="${OBJ_DIR}/${name}.o"
  echo "  ${name}"
  "${HOST_CXX}" "${HOST_FLAGS[@]}" -o "${out}" -c "${ROOT_DIR}/${src}" 2>&1 | head -3
done

# ── Step 4: Pack device kernels into host objects ────────────────────────────
echo ""
echo "=== Step 4: Pack device kernels ==="
for entry in "${KERNELS[@]}"; do
  name="${entry##*:}"
  host_obj="${OBJ_DIR}/${name}_host.o"
  device_obj="${OBJ_DIR}/${name}_device.o"
  packed_obj="${OBJ_DIR}/${name}_packed.o"
  echo "  ${name}"
  "${PACK_KERNEL}" "${host_obj}" "${device_obj}" m200 "${packed_obj}" 2>&1 | head -3
done

# ── Step 5: Link benchmarks ─────────────────────────────────────────────────
echo ""
echo "=== Step 5: Link benchmarks ==="

COMMON_OBJS=()
for src in "${COMMON_SRCS[@]}"; do
  name=$(basename "${src}" .cpp)
  COMMON_OBJS+=("${OBJ_DIR}/${name}.o")
done

for entry in "${BENCHES[@]}"; do
  IFS=':' read -r bench_name main_src kernel_name <<< "${entry}"
  exe="${BIN_DIR}/${bench_name}"
  main_obj="${OBJ_DIR}/${bench_name}_main.o"
  packed_obj="${OBJ_DIR}/${kernel_name}_packed.o"

  echo "  ${bench_name}"
  # Compile main.cpp
  "${HOST_CXX}" "${HOST_FLAGS[@]}" -o "${main_obj}" -c "${ROOT_DIR}/${main_src}" 2>&1 | head -3

  # Link
  "${HOST_CXX}" -o "${exe}" "${main_obj}" "${COMMON_OBJS[@]}" "${packed_obj}" \
    "${HOST_LINK_FLAGS[@]}" 2>&1 | head -5
done

echo ""
echo "=== Build complete ==="
echo "Binaries in ${BIN_DIR}/"
ls -la "${BIN_DIR}/"
