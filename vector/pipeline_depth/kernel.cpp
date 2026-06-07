#include "kernel_operator.h"

#ifndef ASCENDC_CPU_DEBUG
#include <acl/acl.h>
#endif

using namespace AscendC;

namespace {
constexpr uint32_t kElems = 256;
constexpr uint32_t kMaxGap = 8;

template <bool DoCompute>
class VectorPipelineDepthKernel {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t mode) {
    gap_ = mode > kMaxGap ? kMaxGap : mode;
    const uint32_t offset = GetBlockIdx() * kElems * (kMaxGap + 2);
    input_.SetGlobalBuffer((__gm__ float*)input + offset, kElems * (kMaxGap + 2));
    output_.SetGlobalBuffer((__gm__ float*)output + offset, kElems * (kMaxGap + 2));
    pipe_.InitBuffer(src_buf_, kElems * (kMaxGap + 2) * sizeof(float));
    pipe_.InitBuffer(dep_buf_, kElems * sizeof(float));
    pipe_.InitBuffer(tmp_buf_, kElems * kMaxGap * sizeof(float));
  }

  __aicore__ inline void Process(uint32_t repeats) {
    LocalTensor<float> src = src_buf_.Get<float>();
    LocalTensor<float> dep = dep_buf_.Get<float>();
    LocalTensor<float> tmp = tmp_buf_.Get<float>();
    DataCopy(src, input_, kElems * (kMaxGap + 2));
    PipeBarrier<PIPE_MTE2>();
    Add(dep, src, src[kElems], kElems);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < repeats; ++i) {
      if constexpr (DoCompute) {
        Add(dep, dep, src, kElems);
        for (uint32_t g = 0; g < gap_; ++g) {
          Add(tmp[g * kElems], src[(g + 1) * kElems], src[(g + 2) * kElems],
              kElems);
        }
        PipeBarrier<PIPE_V>();
      } else if ((i & 0x7ffu) == 0) {
        { union { uint32_t u; float f; } c; c.u = i; dep.SetValue(0, c.f); }
      }
    }
    DataCopy(output_, dep, kElems);
    PipeBarrier<PIPE_MTE3>();
  }

 private:
  uint32_t gap_ = 0;
  TPipe pipe_;
  TBuf<QuePosition::VECCALC> src_buf_;
  TBuf<QuePosition::VECCALC> dep_buf_;
  TBuf<QuePosition::VECCALC> tmp_buf_;
  GlobalTensor<float> input_;
  GlobalTensor<float> output_;
};
}  // namespace

extern "C" __global__ __aicore__ void vector_pipeline_depth_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  VectorPipelineDepthKernel<true> op;
  op.Init(input, output, mode);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void vector_pipeline_depth_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  VectorPipelineDepthKernel<false> op;
  op.Init(input, output, mode);
  op.Process(repeats);
}

#ifndef ASCENDC_CPU_DEBUG
void vector_pipeline_depth_target_do(uint32_t block_dim, aclrtStream stream,
                                     void* input, void* output,
                                     uint32_t size_bytes, uint32_t repeats,
                                     uint32_t mode) {
  vector_pipeline_depth_target_kernel<<<block_dim, nullptr, stream>>>(
      reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
      size_bytes, repeats, mode);
}

void vector_pipeline_depth_baseline_do(uint32_t block_dim, aclrtStream stream,
                                       void* input, void* output,
                                       uint32_t size_bytes, uint32_t repeats,
                                       uint32_t mode) {
  vector_pipeline_depth_baseline_kernel<<<block_dim, nullptr, stream>>>(
      reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
      size_bytes, repeats, mode);
}
#endif
