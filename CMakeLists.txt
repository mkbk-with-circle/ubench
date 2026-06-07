cmake_minimum_required(VERSION 3.16)
project(ascend_ubench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_CANN_PACKAGE_PATH}" CACHE PATH
    "CANN install path, e.g. /usr/local/Ascend/ascend-toolkit/latest")
set(ASCEND_SOC_VERSION "Ascend310P3" CACHE STRING "Target Ascend SoC version")
set(ASCEND_OPT_LEVEL "O2" CACHE STRING "Ascend kernel optimization level: O0/O2/O3")

if(NOT ASCEND_CANN_PACKAGE_PATH)
  set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

message(STATUS "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "ASCEND_SOC_VERSION=${ASCEND_SOC_VERSION}")
message(STATUS "ASCEND_OPT_LEVEL=${ASCEND_OPT_LEVEL}")

set(ASCEND_INCLUDE_DIR "${ASCEND_CANN_PACKAGE_PATH}/include")
set(UBENCH_HAS_REAL_ACL OFF)
if(EXISTS "${ASCEND_INCLUDE_DIR}/acl/acl.h")
  set(UBENCH_HAS_REAL_ACL ON)
else()
  message(WARNING
    "Could not find ${ASCEND_INCLUDE_DIR}/acl/acl.h. "
    "Building host-only stubs; benchmarks cannot produce valid hardware results.")
endif()

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/common
  ${ASCEND_INCLUDE_DIR}
  ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
)

link_directories(
  ${ASCEND_CANN_PACKAGE_PATH}/lib64
  ${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64
)

set(COMMON_HOST
  common/acl_utils.cpp
  common/bench_utils.cpp
  common/launch_stubs.cpp
)

# The kernel-launch CANN project flow generates launch stubs for __aicore__
# kernels. Different CANN releases use slightly different helper CMake modules,
# so this CMake file keeps host targets explicit and leaves device-kernel
# compilation to scripts/build_kernels.sh.
function(add_ubench name source launch_symbol)
  add_executable(${name} ${source} ${COMMON_HOST})
  target_compile_definitions(${name} PRIVATE
    ASCEND_SOC_VERSION="${ASCEND_SOC_VERSION}"
    ASCEND_OPT_LEVEL="${ASCEND_OPT_LEVEL}"
  )
  if(NOT UBENCH_HAS_REAL_ACL)
    target_compile_definitions(${name} PRIVATE UBENCH_STUB_ACL=1)
  else()
    target_link_libraries(${name} ascendcl)
  endif()
  target_link_libraries(${name} pthread dl)
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
endfunction()

add_ubench(mte_copy_bw mte/copy_bw/main.cpp mte_copy_bw_target)
add_ubench(mte_startup_latency mte/startup_latency/main.cpp mte_startup_latency_target)
add_ubench(mte_granularity mte/granularity/main.cpp mte_granularity_target)

add_ubench(vector_add_latency vector/add_latency/main.cpp vector_add_latency_target)
add_ubench(vector_mul_latency vector/mul_latency/main.cpp vector_mul_latency_target)
add_ubench(vector_throughput vector/throughput/main.cpp vector_throughput_target)
add_ubench(vector_pipeline_depth vector/pipeline_depth/main.cpp vector_pipeline_depth_target)

add_ubench(cube_tile_latency cube/tile_latency/main.cpp cube_tile_latency_target)
add_ubench(cube_throughput cube/throughput/main.cpp cube_throughput_target)
add_ubench(cube_scaling cube/scaling/main.cpp cube_scaling_target)

add_ubench(scalar_arith_latency scalar/arith_latency/main.cpp scalar_arith_latency_target)
add_ubench(scalar_branch_overhead scalar/branch_overhead/main.cpp scalar_branch_overhead_target)
