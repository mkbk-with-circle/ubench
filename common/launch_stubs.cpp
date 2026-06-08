#include "ubench_kernels.h"

#include <iostream>
#include <stdexcept>

namespace {

[[noreturn]] void MissingKernel(const char* name) {
  std::cerr << "missing Ascend C kernel launch stub: " << name << "\n"
            << "Build and link the device kernel object for this benchmark.\n";
  throw std::runtime_error("missing Ascend C kernel launch stub");
}

}  // namespace

#define DEFINE_MISSING_UBENCH_LAUNCH(name)                                    \
  __attribute__((weak)) void name##_do(uint32_t,                              \
                                       aclrtStream,                           \
                                       void*,                                 \
                                       void*,                                 \
                                       uint32_t,                              \
                                       uint32_t,                              \
                                       uint32_t) {                            \
    MissingKernel(#name);                                                     \
  }

DEFINE_MISSING_UBENCH_LAUNCH(mte_copy_bw_target)
DEFINE_MISSING_UBENCH_LAUNCH(mte_copy_bw_baseline)
DEFINE_MISSING_UBENCH_LAUNCH(mte_startup_latency_target)
DEFINE_MISSING_UBENCH_LAUNCH(mte_startup_latency_baseline)
DEFINE_MISSING_UBENCH_LAUNCH(mte_granularity_target)
DEFINE_MISSING_UBENCH_LAUNCH(mte_granularity_baseline)
DEFINE_MISSING_UBENCH_LAUNCH(vector_add_latency_target)
DEFINE_MISSING_UBENCH_LAUNCH(vector_add_latency_baseline)
DEFINE_MISSING_UBENCH_LAUNCH(vector_mul_latency_target)
DEFINE_MISSING_UBENCH_LAUNCH(vector_mul_latency_baseline)
DEFINE_MISSING_UBENCH_LAUNCH(vector_throughput_target)
DEFINE_MISSING_UBENCH_LAUNCH(vector_throughput_baseline)
DEFINE_MISSING_UBENCH_LAUNCH(vector_pipeline_depth_target)
DEFINE_MISSING_UBENCH_LAUNCH(vector_pipeline_depth_baseline)
DEFINE_MISSING_UBENCH_LAUNCH(cube_tile_latency_target)
DEFINE_MISSING_UBENCH_LAUNCH(cube_tile_latency_baseline)
DEFINE_MISSING_UBENCH_LAUNCH(cube_throughput_target)
DEFINE_MISSING_UBENCH_LAUNCH(cube_throughput_baseline)
DEFINE_MISSING_UBENCH_LAUNCH(cube_scaling_target)
DEFINE_MISSING_UBENCH_LAUNCH(cube_scaling_baseline)
DEFINE_MISSING_UBENCH_LAUNCH(scalar_arith_latency_target)
DEFINE_MISSING_UBENCH_LAUNCH(scalar_arith_latency_baseline)
DEFINE_MISSING_UBENCH_LAUNCH(scalar_branch_overhead_target)
DEFINE_MISSING_UBENCH_LAUNCH(scalar_branch_overhead_baseline)
DEFINE_MISSING_UBENCH_LAUNCH(scalar_throughput_target)
DEFINE_MISSING_UBENCH_LAUNCH(scalar_throughput_baseline)
DEFINE_MISSING_UBENCH_LAUNCH(scalar_mem_latency_target)
DEFINE_MISSING_UBENCH_LAUNCH(scalar_mem_latency_baseline)
DEFINE_MISSING_UBENCH_LAUNCH(vector_reg_latency_target)
DEFINE_MISSING_UBENCH_LAUNCH(vector_reg_latency_baseline)
DEFINE_MISSING_UBENCH_LAUNCH(mte_write_bw_target)
DEFINE_MISSING_UBENCH_LAUNCH(mte_write_bw_baseline)
DEFINE_MISSING_UBENCH_LAUNCH(mte_hbm_latency_target)
DEFINE_MISSING_UBENCH_LAUNCH(mte_hbm_latency_baseline)
DEFINE_MISSING_UBENCH_LAUNCH(mte_buffer_capacity_target)
DEFINE_MISSING_UBENCH_LAUNCH(mte_buffer_capacity_baseline)
