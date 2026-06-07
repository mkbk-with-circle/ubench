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
    if constexpr (DoMatmul) {
      // Initialize Matmul with tiling for 16x16x16
      TCubeTiling tiling;
      tiling.usedCoreNum = 1;
      tiling.M = kM;
      tiling.N = kN;
      tiling.Ka = kK;
      tiling.Kb = kK;
      tiling.singleCoreM = kM;
      tiling.singleCoreN = kN;
      tiling.singleCoreK = kK;
      tiling.baseM = kM;
      tiling.baseN = kN;
      tiling.baseK = kK;
      tiling.depthA1 = 1;
      tiling.depthB1 = 1;
      tiling.stepM = 1;
      tiling.stepN = 1;
      tiling.isBias = 0;
      tiling.transLength = 0;
      tiling.iterateOrder = 0;
      tiling.shareMode = 0;
      tiling.shareL1Size = 0;
      tiling.shareL0CSize = 0;
      tiling.shareUbSize = 0;
      tiling.batchM = 1;
      tiling.batchN = 1;
      tiling.singleBatchM = kM;
      tiling.singleBatchN = kN;
      tiling.stepKa = 1;
      tiling.stepKb = 1;
      tiling.depthAL1CacheUB = 0;
      tiling.depthBL1CacheUB = 0;
      mm_.Init(&tiling);
      mm_.SetOrgShape(kM, kN, kK);
      for (uint32_t i = 0; i < repeats; ++i) {
        mm_.SetTensorA(a_);
        mm_.SetTensorB(b_);
        mm_.IterateAll(c_);
      }
    } else {
      for (uint32_t i = 0; i < repeats; ++i) {
        if ((i & 0x7ffu) == 0) {
          { union { uint32_t u; float f; } c; c.u = i; c_.SetValue(0, c.f); }
        }
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
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  RunCubeKernel<true>(input, output, repeats, mode);
}

extern "C" __global__ __aicore__ void cube_tile_latency_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  RunCubeKernel<false>(input, output, repeats, mode);
}

extern "C" __global__ __aicore__ void cube_throughput_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  RunCubeKernel<true>(input, output, repeats, mode);
}

extern "C" __global__ __aicore__ void cube_throughput_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  RunCubeKernel<false>(input, output, repeats, mode);
}

extern "C" __global__ __aicore__ void cube_scaling_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  RunCubeKernel<true>(input, output, repeats, mode);
}

extern "C" __global__ __aicore__ void cube_scaling_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
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
