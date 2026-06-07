#include "kernel_operator.h"

#ifndef ASCENDC_CPU_DEBUG
#include <acl/acl.h>
#endif

using namespace AscendC;

namespace {
constexpr uint32_t kElems = 256;

template <bool DoMul>
class VectorMulLatencyKernel {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
    const uint32_t offset = GetBlockIdx() * kElems;
    input_.SetGlobalBuffer((__gm__ float*)input + offset, kElems);
    output_.SetGlobalBuffer((__gm__ float*)output + offset, kElems);
    pipe_.InitBuffer(a_buf_, kElems * sizeof(float));
    pipe_.InitBuffer(acc_buf_, kElems * sizeof(float));
  }

  __aicore__ inline void Process(uint32_t repeats) {
    LocalTensor<float> a = a_buf_.Get<float>();
    LocalTensor<float> acc = acc_buf_.Get<float>();
    DataCopy(a, input_, kElems);
    DataCopy(acc, input_, kElems);
    PipeBarrier<PIPE_MTE2>();
    for (uint32_t i = 0; i < repeats; ++i) {
      if constexpr (DoMul) {
        Mul(acc, acc, a, kElems);
        PipeBarrier<PIPE_V>();
      } else if ((i & 0x7ffu) == 0) {
        acc.SetValue(0, static_cast<float>(i + 1));
      }
    }
    DataCopy(output_, acc, kElems);
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

extern "C" __global__ __aicore__ void vector_mul_latency_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t, uint32_t repeats, uint32_t) {
  VectorMulLatencyKernel<true> op;
  op.Init(input, output);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void vector_mul_latency_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t, uint32_t repeats, uint32_t) {
  VectorMulLatencyKernel<false> op;
  op.Init(input, output);
  op.Process(repeats);
}

#ifndef ASCENDC_CPU_DEBUG
void vector_mul_latency_target_do(uint32_t block_dim, aclrtStream stream,
                                  void* input, void* output,
                                  uint32_t size_bytes, uint32_t repeats,
                                  uint32_t mode) {
  vector_mul_latency_target_kernel<<<block_dim, nullptr, stream>>>(
      reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
      size_bytes, repeats, mode);
}

void vector_mul_latency_baseline_do(uint32_t block_dim, aclrtStream stream,
                                    void* input, void* output,
                                    uint32_t size_bytes, uint32_t repeats,
                                    uint32_t mode) {
  vector_mul_latency_baseline_kernel<<<block_dim, nullptr, stream>>>(
      reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
      size_bytes, repeats, mode);
}
#endif
