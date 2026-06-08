cmake_minimum_required(VERSION 3.16)
project(ascend_ubench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── CANN paths ──────────────────────────────────────────────────────────────
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_CANN_PACKAGE_PATH}" CACHE PATH
    "CANN install path")
if(NOT ASCEND_CANN_PACKAGE_PATH)
  set(ASCEND_CANN_PACKAGE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cann_wrapper")
endif()

set(ASCEND_SOC_VERSION "$ENV{ASCEND_SOC_VERSION}" CACHE STRING "Target SoC")
if(NOT ASCEND_SOC_VERSION)
  set(ASCEND_SOC_VERSION "ascend910b")
endif()

set(ASCEND_OPT_LEVEL "$ENV{ASCEND_OPT_LEVEL}" CACHE STRING "Opt level")
if(NOT ASCEND_OPT_LEVEL)
  set(ASCEND_OPT_LEVEL "O2")
endif()

message(STATUS "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "ASCEND_SOC_VERSION=${ASCEND_SOC_VERSION}")

# ── Ascend C kernel build system ────────────────────────────────────────────
# Use CANN's built-in ascendc cmake modules (CANN 9.0.0 structure)
set(ASCENDC_KERNEL_CMAKE_DIR "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake")
set(CMAKE_MODULE_PATH "${ASCENDC_KERNEL_CMAKE_DIR}" ${CMAKE_MODULE_PATH})

# SOC_VERSION is required by host_config.cmake
set(SOC_VERSION ${ASCEND_SOC_VERSION})

message(STATUS "CMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}")
message(STATUS "ASCENDC_KERNEL_CMAKE_DIR=${ASCENDC_KERNEL_CMAKE_DIR}")
include(ascendc)

# ── Common host sources ─────────────────────────────────────────────────────
set(COMMON_HOST
  common/acl_utils.cpp
  common/bench_utils.cpp
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/common
  ${ASCEND_CANN_PACKAGE_PATH}/include
  ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include/adv_api
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include/basic_api
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include
)

link_directories(
  ${ASCEND_CANN_PACKAGE_PATH}/lib64
  ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/lib64
  ${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64
  /usr/local/Ascend/driver/lib64
  /usr/local/Ascend/driver/lib64/common
  /usr/local/Ascend/driver/lib64/driver
)

# ── Build Ascend C kernel libraries ─────────────────────────────────────────
# Each kernel.cpp is compiled with the Ascend C toolchain, producing a library
# that contains both device code and host launch stubs.

set(KERNEL_SOURCES
  mte_copy_bw          mte/copy_bw/kernel.cpp
  mte_startup_latency  mte/startup_latency/kernel.cpp
  mte_granularity      mte/granularity/kernel.cpp
  mte_write_bw         mte/write_bw/kernel.cpp
  mte_hbm_latency      mte/hbm_latency/kernel.cpp
  mte_buffer_capacity  mte/buffer_capacity/kernel.cpp
  vector_add_latency   vector/add_latency/kernel.cpp
  vector_mul_latency   vector/mul_latency/kernel.cpp
  vector_throughput    vector/throughput/kernel.cpp
  vector_pipeline_depth vector/pipeline_depth/kernel.cpp
  vector_reg_latency   vector/reg_latency/kernel.cpp
  cube_matmul          cube/cube_matmul_kernel.cpp
  scalar               scalar/scalar_kernel.cpp
)

list(LENGTH KERNEL_SOURCES _ks_len)
math(EXPR _ks_last "${_ks_len} - 1")
foreach(_idx RANGE 0 ${_ks_last} 2)
  math(EXPR _src_idx "${_idx} + 1")
  list(GET KERNEL_SOURCES ${_idx} _kname)
  list(GET KERNEL_SOURCES ${_src_idx} _ksrc)
  set(_target "${_kname}_kernels")
  ascendc_library(${_target} STATIC ${_ksrc})
  # Add ACL include path for the precompile step
  set_target_properties(${_target}_interface PROPERTIES
    INCLUDES "${ASCEND_CANN_PACKAGE_PATH}/include;${CMAKE_CURRENT_SOURCE_DIR}/local_include"
  )
endforeach()

# ── Build host executables ──────────────────────────────────────────────────
function(add_ubench name source kernel_lib)
  add_executable(${name} ${source} ${COMMON_HOST})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${ASCEND_CANN_PACKAGE_PATH}/include
  )
  target_compile_definitions(${name} PRIVATE
    ASCEND_SOC_VERSION="${ASCEND_SOC_VERSION}"
    ASCEND_OPT_LEVEL="${ASCEND_OPT_LEVEL}"
  )
  # CANN 9.0.0 has complex circular dependencies between its shared libraries.
  # --allow-shlib-undefined lets the runtime resolve them at load time.
  target_link_libraries(${name} PRIVATE ${kernel_lib})
  target_link_libraries(${name} PRIVATE
    -Wl,--allow-shlib-undefined
    -Wl,--start-group
    ascendcl runtime tiling_api register platform error_manager profapi
    ge_common ge_common_base gert unified_dlog ascend_dump mmpa c_sec
    -Wl,--end-group
    pthread dl)
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
endfunction()

add_ubench(mte_copy_bw           mte/copy_bw/main.cpp           mte_copy_bw_kernels)
add_ubench(mte_startup_latency   mte/startup_latency/main.cpp   mte_startup_latency_kernels)
add_ubench(mte_granularity       mte/granularity/main.cpp       mte_granularity_kernels)
add_ubench(mte_write_bw          mte/write_bw/main.cpp          mte_write_bw_kernels)
add_ubench(mte_hbm_latency       mte/hbm_latency/main.cpp       mte_hbm_latency_kernels)
add_ubench(mte_buffer_capacity   mte/buffer_capacity/main.cpp   mte_buffer_capacity_kernels)
add_ubench(vector_add_latency    vector/add_latency/main.cpp    vector_add_latency_kernels)
add_ubench(vector_mul_latency    vector/mul_latency/main.cpp    vector_mul_latency_kernels)
add_ubench(vector_throughput     vector/throughput/main.cpp     vector_throughput_kernels)
add_ubench(vector_pipeline_depth vector/pipeline_depth/main.cpp vector_pipeline_depth_kernels)
add_ubench(vector_reg_latency    vector/reg_latency/main.cpp    vector_reg_latency_kernels)
add_ubench(cube_tile_latency     cube/tile_latency/main.cpp     cube_matmul_kernels)
add_ubench(cube_throughput       cube/throughput/main.cpp       cube_matmul_kernels)
add_ubench(cube_scaling          cube/scaling/main.cpp          cube_matmul_kernels)
add_ubench(scalar_arith_latency  scalar/arith_latency/main.cpp  scalar_kernels)
add_ubench(scalar_branch_overhead scalar/branch_overhead/main.cpp scalar_kernels)
add_ubench(scalar_throughput     scalar/throughput/main.cpp     scalar_kernels)
add_ubench(scalar_mem_latency    scalar/mem_latency/main.cpp    scalar_kernels)
