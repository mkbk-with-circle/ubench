#include "kernel_operator.h"

#ifndef ASCENDC_CPU_DEBUG
#include <acl/acl.h>
#endif

using namespace AscendC;

namespace {
constexpr uint32_t kM = 128;
constexpr uint32_t kN = 128;
constexpr uint32_t kK = 128;
constexpr uint32_t kAElems = kM * kK;
constexpr uint32_t kBElems = kK * kN;
constexpr uint32_t kCElems = kM * kN;

// Placeholder kernel: Cube benchmarks require AIC compilation
// which has registration issues on CANN 9.0 / 910B (error 107000/507000)
// These kernels are compiled as AIV-only and produce dummy results.

template <bool DoCompute>
class CubePlaceholderKernel {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
    output_.SetGlobalBuffer((__gm__ float*)output, kCElems);
  }

  __aicore__ inline void Process(uint32_t repeats) {
    for (uint32_t i = 0; i < repeats; ++i) {
      if ((i & 0x7ffu) == 0) {
        { union { uint32_t u; float f; } c; c.u = i; output_.SetValue(0, c.f); }
      }
    }
  }

 private:
  GlobalTensor<float> output_;
};
}  // namespace

extern "C" __global__ __aicore__ void cube_tile_latency_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  CubePlaceholderKernel<true> op;
  op.Init(input, output);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void cube_tile_latency_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  CubePlaceholderKernel<false> op;
  op.Init(input, output);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void cube_throughput_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  CubePlaceholderKernel<true> op;
  op.Init(input, output);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void cube_throughput_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  CubePlaceholderKernel<false> op;
  op.Init(input, output);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void cube_scaling_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  CubePlaceholderKernel<true> op;
  op.Init(input, output);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void cube_scaling_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  CubePlaceholderKernel<false> op;
  op.Init(input, output);
  op.Process(repeats);
}

#ifndef ASCENDC_CPU_DEBUG
#define DEFINE_CUBE_DO(name)                                                   \
  void name##_do(uint32_t block_dim, aclrtStream stream, void* input,          \
                 void* output, uint32_t size_bytes, uint32_t repeats,          \
                 uint32_t mode) {                                              \
    name##_kernel<<<block_dim, nullptr, stream>>>(                             \
        reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),   \
        size_bytes, repeats, mode);                                            \
  }

DEFINE_CUBE_DO(cube_tile_latency_target)
DEFINE_CUBE_DO(cube_tile_latency_baseline)
DEFINE_CUBE_DO(cube_throughput_target)
DEFINE_CUBE_DO(cube_throughput_baseline)
DEFINE_CUBE_DO(cube_scaling_target)
DEFINE_CUBE_DO(cube_scaling_baseline)
#endif
