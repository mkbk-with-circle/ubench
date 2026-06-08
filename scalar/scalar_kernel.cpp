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
#endif
