#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#ifndef ASCENDC_CPU_DEBUG
#include <acl/acl.h>
#endif

using namespace AscendC;

namespace {
constexpr uint32_t kM = 16;
constexpr uint32_t kN = 16;
constexpr uint32_t kK = 16;
constexpr uint32_t kAElems = kM * kK;
constexpr uint32_t kBElems = kK * kN;
constexpr uint32_t kCElems = kM * kN;

using AType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

template <bool DoMatmul>
class CubeMatmulKernel {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t mode) {
    const uint32_t tile_id = GetBlockIdx() * (mode == 0 ? 1 : mode);
    const uint32_t a_offset = tile_id * (kAElems + kBElems);
    const uint32_t b_offset = a_offset + kAElems;
    const uint32_t c_offset = GetBlockIdx() * kCElems;
    a_.SetGlobalBuffer((__gm__ half*)input + a_offset, kAElems);
    b_.SetGlobalBuffer((__gm__ half*)input + b_offset, kBElems);
    c_.SetGlobalBuffer((__gm__ float*)output + c_offset, kCElems);
  }

  __aicore__ inline void Process(uint32_t repeats) {
    for (uint32_t i = 0; i < repeats; ++i) {
      if constexpr (DoMatmul) {
        mm_.SetOrgShape(kM, kN, kK);
        mm_.SetTensorA(a_);
        mm_.SetTensorB(b_);
        mm_.IterateAll(c_);
      } else if ((i & 0x7ffu) == 0) {
        // Keep the loop observable without issuing Cube work.
        c_.SetValue(0, static_cast<float>(i));
      }
    }
  }

 private:
  GlobalTensor<half> a_;
  GlobalTensor<half> b_;
  GlobalTensor<float> c_;
  Matmul<AType, BType, CType, BiasType> mm_;
};

template <bool DoMatmul>
__aicore__ inline void RunCubeKernel(GM_ADDR input, GM_ADDR output,
                                     uint32_t repeats, uint32_t mode) {
  CubeMatmulKernel<DoMatmul> op;
  op.Init(input, output, mode);
  op.Process(repeats);
}
}  // namespace

extern "C" __global__ __aicore__ void cube_tile_latency_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t, uint32_t repeats, uint32_t mode) {
  RunCubeKernel<true>(input, output, repeats, mode);
}

extern "C" __global__ __aicore__ void cube_tile_latency_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t, uint32_t repeats, uint32_t mode) {
  RunCubeKernel<false>(input, output, repeats, mode);
}

extern "C" __global__ __aicore__ void cube_throughput_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t, uint32_t repeats, uint32_t mode) {
  RunCubeKernel<true>(input, output, repeats, mode);
}

extern "C" __global__ __aicore__ void cube_throughput_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t, uint32_t repeats, uint32_t mode) {
  RunCubeKernel<false>(input, output, repeats, mode);
}

extern "C" __global__ __aicore__ void cube_scaling_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t, uint32_t repeats, uint32_t mode) {
  RunCubeKernel<true>(input, output, repeats, mode);
}

extern "C" __global__ __aicore__ void cube_scaling_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t, uint32_t repeats, uint32_t mode) {
  RunCubeKernel<false>(input, output, repeats, mode);
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
