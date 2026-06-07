#include "kernel_operator.h"

#ifndef ASCENDC_CPU_DEBUG
#include <acl/acl.h>
#endif

using namespace AscendC;

namespace {
constexpr uint32_t kElems = 256;
constexpr uint32_t kLanes = 4;

template <bool DoCompute>
class VectorThroughputKernel {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
    const uint32_t offset = GetBlockIdx() * kElems * kLanes;
    input_.SetGlobalBuffer((__gm__ float*)input + offset, kElems * kLanes);
    output_.SetGlobalBuffer((__gm__ float*)output + offset, kElems * kLanes);
    pipe_.InitBuffer(a_buf_, kElems * kLanes * sizeof(float));
    pipe_.InitBuffer(b_buf_, kElems * kLanes * sizeof(float));
    pipe_.InitBuffer(out_buf_, kElems * kLanes * sizeof(float));
  }

  __aicore__ inline void Process(uint32_t repeats) {
    LocalTensor<float> a = a_buf_.Get<float>();
    LocalTensor<float> b = b_buf_.Get<float>();
    LocalTensor<float> out = out_buf_.Get<float>();
    DataCopy(a, input_, kElems * kLanes);
    DataCopy(b, input_, kElems * kLanes);
    PipeBarrier<PIPE_MTE2>();
    for (uint32_t i = 0; i < repeats; ++i) {
      if constexpr (DoCompute) {
        Add(out[0 * kElems], a[0 * kElems], b[0 * kElems], kElems);
        Add(out[1 * kElems], a[1 * kElems], b[1 * kElems], kElems);
        Add(out[2 * kElems], a[2 * kElems], b[2 * kElems], kElems);
        Add(out[3 * kElems], a[3 * kElems], b[3 * kElems], kElems);
      } else if ((i & 0x7ffu) == 0) {
        { union { uint32_t u; float f; } c; c.u = i; out.SetValue(0, c.f); }
      }
    }
    PipeBarrier<PIPE_V>();
    DataCopy(output_, out, kElems * kLanes);
    PipeBarrier<PIPE_MTE3>();
  }

 private:
  TPipe pipe_;
  TBuf<QuePosition::VECCALC> a_buf_;
  TBuf<QuePosition::VECCALC> b_buf_;
  TBuf<QuePosition::VECCALC> out_buf_;
  GlobalTensor<float> input_;
  GlobalTensor<float> output_;
};
}  // namespace

extern "C" __global__ __aicore__ void vector_throughput_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  VectorThroughputKernel<true> op;
  op.Init(input, output);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void vector_throughput_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  VectorThroughputKernel<false> op;
  op.Init(input, output);
  op.Process(repeats);
}

#ifndef ASCENDC_CPU_DEBUG
void vector_throughput_target_do(uint32_t block_dim, aclrtStream stream,
                                 void* input, void* output,
                                 uint32_t size_bytes, uint32_t repeats,
                                 uint32_t mode) {
  vector_throughput_target_kernel<<<block_dim, nullptr, stream>>>(
      reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
      size_bytes, repeats, mode);
}

void vector_throughput_baseline_do(uint32_t block_dim, aclrtStream stream,
                                   void* input, void* output,
                                   uint32_t size_bytes, uint32_t repeats,
                                   uint32_t mode) {
  vector_throughput_baseline_kernel<<<block_dim, nullptr, stream>>>(
      reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
      size_bytes, repeats, mode);
}
#endif
