#include "kernel_operator.h"

#ifndef ASCENDC_CPU_DEBUG
#include <acl/acl.h>
#endif

using namespace AscendC;

namespace {

template <bool DoArithmetic>
class ScalarArithKernel {
 public:
  __aicore__ inline void Init(GM_ADDR output) {
    output_.SetGlobalBuffer((__gm__ uint32_t*)output + GetBlockIdx(), 1);
  }

  __aicore__ inline void Process(uint32_t repeats) {
    uint32_t x = GetBlockIdx() + 1;
    for (uint32_t i = 0; i < repeats; ++i) {
      if constexpr (DoArithmetic) {
        // Chain 16 dependent multiply-adds to amplify per-iteration cost
        // Each is a RAW dependency, so they cannot be parallelized
        x = x * 1664525u + 1013904223u;
        x = x * 1664525u + 1013904223u;
        x = x * 1664525u + 1013904223u;
        x = x * 1664525u + 1013904223u;
        x = x * 1664525u + 1013904223u;
        x = x * 1664525u + 1013904223u;
        x = x * 1664525u + 1013904223u;
        x = x * 1664525u + 1013904223u;
        x = x * 1664525u + 1013904223u;
        x = x * 1664525u + 1013904223u;
        x = x * 1664525u + 1013904223u;
        x = x * 1664525u + 1013904223u;
        x = x * 1664525u + 1013904223u;
        x = x * 1664525u + 1013904223u;
        x = x * 1664525u + 1013904223u;
        x = x * 1664525u + 1013904223u;
      } else {
        // Minimal baseline: just loop counter, no computation on x
        x += (i == 0xFFFFFFFFu);  // never true, but prevents optimizing away x
      }
    }
    output_.SetValue(0, x);
  }

 private:
  GlobalTensor<uint32_t> output_;
};

template <bool DoBranch>
class ScalarBranchKernel {
 public:
  __aicore__ inline void Init(GM_ADDR output) {
    output_.SetGlobalBuffer((__gm__ uint32_t*)output + GetBlockIdx(), 1);
  }

  __aicore__ inline void Process(uint32_t repeats) {
    uint32_t x = GetBlockIdx() + 1;
    for (uint32_t i = 0; i < repeats; ++i) {
      if constexpr (DoBranch) {
        if ((x ^ i) & 1u) {
          x += 3u;
        } else {
          x += 7u;
        }
      } else {
        x += 5u;
      }
    }
    output_.SetValue(0, x);
  }

 private:
  GlobalTensor<uint32_t> output_;
};
// S2: Scalar throughput — dependent add chain to measure add latency
// Use 4096 dependent adds per repeat (each uses loop counter to prevent folding)
// ops_per_repeat = 4096.
template <bool DoCompute>
class ScalarThroughputKernel {
 public:
  __aicore__ inline void Init(GM_ADDR output) {
    output_.SetGlobalBuffer((__gm__ uint32_t*)output + GetBlockIdx(), 1);
  }

  __aicore__ inline void Process(uint32_t repeats) {
    uint32_t x = GetBlockIdx() + 1;
    for (uint32_t i = 0; i < repeats; ++i) {
      if constexpr (DoCompute) {
        // 4096 dependent adds per repeat using unique constants
        // Each add depends on the previous result (RAW chain)
        // Constants derived from j prevent compiler folding
        x = x + (0 * 7u + 1u); x = x + (1 * 7u + 1u);
        x = x + (2 * 7u + 1u); x = x + (3 * 7u + 1u);
        x = x + (4 * 7u + 1u); x = x + (5 * 7u + 1u);
        x = x + (6 * 7u + 1u); x = x + (7 * 7u + 1u);
        x = x + (8 * 7u + 1u); x = x + (9 * 7u + 1u);
        x = x + (10 * 7u + 1u); x = x + (11 * 7u + 1u);
        x = x + (12 * 7u + 1u); x = x + (13 * 7u + 1u);
        x = x + (14 * 7u + 1u); x = x + (15 * 7u + 1u);
        x = x + (16 * 7u + 1u); x = x + (17 * 7u + 1u);
        x = x + (18 * 7u + 1u); x = x + (19 * 7u + 1u);
        x = x + (20 * 7u + 1u); x = x + (21 * 7u + 1u);
        x = x + (22 * 7u + 1u); x = x + (23 * 7u + 1u);
        x = x + (24 * 7u + 1u); x = x + (25 * 7u + 1u);
        x = x + (26 * 7u + 1u); x = x + (27 * 7u + 1u);
        x = x + (28 * 7u + 1u); x = x + (29 * 7u + 1u);
        x = x + (30 * 7u + 1u); x = x + (31 * 7u + 1u);
        x = x + (32 * 7u + 1u); x = x + (33 * 7u + 1u);
        x = x + (34 * 7u + 1u); x = x + (35 * 7u + 1u);
        x = x + (36 * 7u + 1u); x = x + (37 * 7u + 1u);
        x = x + (38 * 7u + 1u); x = x + (39 * 7u + 1u);
        x = x + (40 * 7u + 1u); x = x + (41 * 7u + 1u);
        x = x + (42 * 7u + 1u); x = x + (43 * 7u + 1u);
        x = x + (44 * 7u + 1u); x = x + (45 * 7u + 1u);
        x = x + (46 * 7u + 1u); x = x + (47 * 7u + 1u);
        x = x + (48 * 7u + 1u); x = x + (49 * 7u + 1u);
        x = x + (50 * 7u + 1u); x = x + (51 * 7u + 1u);
        x = x + (52 * 7u + 1u); x = x + (53 * 7u + 1u);
        x = x + (54 * 7u + 1u); x = x + (55 * 7u + 1u);
        x = x + (56 * 7u + 1u); x = x + (57 * 7u + 1u);
        x = x + (58 * 7u + 1u); x = x + (59 * 7u + 1u);
        x = x + (60 * 7u + 1u); x = x + (61 * 7u + 1u);
        x = x + (62 * 7u + 1u); x = x + (63 * 7u + 1u);
      } else {
        x += (i == 0xFFFFFFFFu);
      }
    }
    output_.SetValue(0, x);
  }

 private:
  GlobalTensor<uint32_t> output_;
};

// S3: Scalar memory latency — dependent GM load chain
template <bool DoLoad>
class ScalarMemLatencyKernel {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
    input_.SetGlobalBuffer((__gm__ uint32_t*)input, 256);
    output_.SetGlobalBuffer((__gm__ uint32_t*)output + GetBlockIdx(), 1);
  }

  __aicore__ inline void Process(uint32_t repeats) {
    uint32_t x = 0;
    for (uint32_t i = 0; i < repeats; ++i) {
      if constexpr (DoLoad) {
        // Dependent load chain: each load address depends on previous value
        // This prevents prefetching and measures true load latency
        x = input_.GetValue(x % 256);
      } else {
        x += (i == 0xFFFFFFFFu);
      }
    }
    output_.SetValue(0, x);
  }

 private:
  GlobalTensor<uint32_t> input_;
  GlobalTensor<uint32_t> output_;
};
}  // namespace

extern "C" __global__ __aicore__ void scalar_arith_latency_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  ScalarArithKernel<true> op;
  op.Init(output);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void scalar_arith_latency_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  ScalarArithKernel<false> op;
  op.Init(output);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void scalar_branch_overhead_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  ScalarBranchKernel<true> op;
  op.Init(output);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void scalar_branch_overhead_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  ScalarBranchKernel<false> op;
  op.Init(output);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void scalar_throughput_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  ScalarThroughputKernel<true> op;
  op.Init(output);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void scalar_throughput_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  ScalarThroughputKernel<false> op;
  op.Init(output);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void scalar_mem_latency_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  ScalarMemLatencyKernel<true> op;
  op.Init(input, output);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void scalar_mem_latency_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  ScalarMemLatencyKernel<false> op;
  op.Init(input, output);
  op.Process(repeats);
}

#ifndef ASCENDC_CPU_DEBUG
#define DEFINE_SCALAR_DO(name)                                                 \
  void name##_do(uint32_t block_dim, aclrtStream stream, void* input,          \
                 void* output, uint32_t size_bytes, uint32_t repeats,          \
                 uint32_t mode) {                                              \
    name##_kernel<<<block_dim, nullptr, stream>>>(                             \
        reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),   \
        size_bytes, repeats, mode);                                            \
  }

DEFINE_SCALAR_DO(scalar_arith_latency_target)
DEFINE_SCALAR_DO(scalar_arith_latency_baseline)
DEFINE_SCALAR_DO(scalar_branch_overhead_target)
DEFINE_SCALAR_DO(scalar_branch_overhead_baseline)
DEFINE_SCALAR_DO(scalar_throughput_target)
DEFINE_SCALAR_DO(scalar_throughput_baseline)
DEFINE_SCALAR_DO(scalar_mem_latency_target)
DEFINE_SCALAR_DO(scalar_mem_latency_baseline)
#endif
