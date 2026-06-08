#include "kernel_operator.h"

#ifndef ASCENDC_CPU_DEBUG
#include <acl/acl.h>
#endif

using namespace AscendC;

namespace {
// V5: Vector register read/write latency
// Use 1-element RAW dependency chain to measure register-level latency
constexpr uint32_t kElems = 1;
constexpr uint32_t kBufElems = 256;  // Buffer for DataCopy

template <bool DoCompute>
class VectorRegLatencyKernel {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
    const uint32_t offset = GetBlockIdx() * kBufElems;
    input_.SetGlobalBuffer((__gm__ float*)input + offset, kBufElems);
    output_.SetGlobalBuffer((__gm__ float*)output + offset, kBufElems);
    pipe_.InitBuffer(a_buf_, kBufElems * sizeof(float));
    pipe_.InitBuffer(acc_buf_, kBufElems * sizeof(float));
  }

  __aicore__ inline void Process(uint32_t repeats) {
    LocalTensor<float> a = a_buf_.Get<float>();
    LocalTensor<float> acc = acc_buf_.Get<float>();
    DataCopy(a, input_, kBufElems);
    DataCopy(acc, input_, kBufElems);
    PipeBarrier<PIPE_MTE2>();
    for (uint32_t i = 0; i < repeats; ++i) {
      if constexpr (DoCompute) {
        // Single-element RAW dependency: measures register read-after-write latency
        Add(acc, acc, a, kElems);
        PipeBarrier<PIPE_V>();
      } else if ((i & 0x7ffu) == 0) {
        { union { uint32_t u; float f; } c; c.u = i; acc.SetValue(0, c.f); }
      }
    }
    DataCopy(output_, acc, kBufElems);
    PipeBarrier<PIPE_MTE3>();
  }

 private:
  TPipe pipe_;
  TBuf<QuePosition::VECCALC> a_buf_;
  TBuf<QuePosition::VECCALC> acc_buf_;
  GlobalTensor<float> input_;
  GlobalTensor<float> output_;
};
}  // namespace

extern "C" __global__ __aicore__ void vector_reg_latency_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  VectorRegLatencyKernel<true> op;
  op.Init(input, output);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void vector_reg_latency_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  VectorRegLatencyKernel<false> op;
  op.Init(input, output);
  op.Process(repeats);
}

#ifndef ASCENDC_CPU_DEBUG
void vector_reg_latency_target_do(uint32_t block_dim, aclrtStream stream,
                                   void* input, void* output,
                                   uint32_t size_bytes, uint32_t repeats,
                                   uint32_t mode) {
  vector_reg_latency_target_kernel<<<block_dim, nullptr, stream>>>(
      reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
      size_bytes, repeats, mode);
}

void vector_reg_latency_baseline_do(uint32_t block_dim, aclrtStream stream,
                                     void* input, void* output,
                                     uint32_t size_bytes, uint32_t repeats,
                                     uint32_t mode) {
  vector_reg_latency_baseline_kernel<<<block_dim, nullptr, stream>>>(
      reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
      size_bytes, repeats, mode);
}
#endif
