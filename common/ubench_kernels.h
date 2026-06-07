#pragma once

#include "acl_compat.h"

#include <cstdint>

using UbenchLaunchFn = void (*)(uint32_t block_dim,
                                aclrtStream stream,
                                void* input,
                                void* output,
                                uint32_t size_bytes,
                                uint32_t repeats,
                                uint32_t mode);

#define DECLARE_UBENCH_LAUNCH(name)                                           \
  void name##_do(uint32_t block_dim,                                          \
                 aclrtStream stream,                                          \
                 void* input,                                                 \
                 void* output,                                                \
                 uint32_t size_bytes,                                         \
                 uint32_t repeats,                                            \
                 uint32_t mode)

DECLARE_UBENCH_LAUNCH(mte_copy_bw_target);
DECLARE_UBENCH_LAUNCH(mte_copy_bw_baseline);
DECLARE_UBENCH_LAUNCH(mte_startup_latency_target);
DECLARE_UBENCH_LAUNCH(mte_startup_latency_baseline);
DECLARE_UBENCH_LAUNCH(mte_granularity_target);
DECLARE_UBENCH_LAUNCH(mte_granularity_baseline);
DECLARE_UBENCH_LAUNCH(vector_add_latency_target);
DECLARE_UBENCH_LAUNCH(vector_add_latency_baseline);
DECLARE_UBENCH_LAUNCH(vector_mul_latency_target);
DECLARE_UBENCH_LAUNCH(vector_mul_latency_baseline);
DECLARE_UBENCH_LAUNCH(vector_throughput_target);
DECLARE_UBENCH_LAUNCH(vector_throughput_baseline);
DECLARE_UBENCH_LAUNCH(vector_pipeline_depth_target);
DECLARE_UBENCH_LAUNCH(vector_pipeline_depth_baseline);
DECLARE_UBENCH_LAUNCH(cube_tile_latency_target);
DECLARE_UBENCH_LAUNCH(cube_tile_latency_baseline);
DECLARE_UBENCH_LAUNCH(cube_throughput_target);
DECLARE_UBENCH_LAUNCH(cube_throughput_baseline);
DECLARE_UBENCH_LAUNCH(cube_scaling_target);
DECLARE_UBENCH_LAUNCH(cube_scaling_baseline);
DECLARE_UBENCH_LAUNCH(scalar_arith_latency_target);
DECLARE_UBENCH_LAUNCH(scalar_arith_latency_baseline);
DECLARE_UBENCH_LAUNCH(scalar_branch_overhead_target);
DECLARE_UBENCH_LAUNCH(scalar_branch_overhead_baseline);
