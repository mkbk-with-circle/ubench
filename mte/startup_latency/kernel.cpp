#include "kernel_operator.h"

#ifndef ASCENDC_CPU_DEBUG
#include <acl/acl.h>
#endif

using namespace AscendC;

namespace {
constexpr uint32_t kSmallCopyBytes = 32;

class MteStartupKernel {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
    const uint32_t offset = GetBlockIdx() * kSmallCopyBytes;
    input_.SetGlobalBuffer((__gm__ uint8_t*)input + offset, kSmallCopyBytes);
    output_.SetGlobalBuffer((__gm__ uint8_t*)output + offset, kSmallCopyBytes);
    pipe_.InitBuffer(tile_buf_, kSmallCopyBytes);
  }

  __aicore__ inline void Process(uint32_t repeats) {
    LocalTensor<uint8_t> tile = tile_buf_.Get<uint8_t>();
    for (uint32_t i = 0; i < repeats; ++i) {
      DataCopy(tile, input_, kSmallCopyBytes);
      PipeBarrier<PIPE_MTE2>();
      DataCopy(output_, tile, kSmallCopyBytes);
      PipeBarrier<PIPE_MTE3>();
    }
  }

 private:
  TPipe pipe_;
  TBuf<QuePosition::VECCALC> tile_buf_;
  GlobalTensor<uint8_t> input_;
  GlobalTensor<uint8_t> output_;
};

class MteStartupBaselineKernel {
 public:
  __aicore__ inline void Init(GM_ADDR output) {
    const uint32_t offset = GetBlockIdx() * kSmallCopyBytes;
    output_.SetGlobalBuffer((__gm__ uint8_t*)output + offset, kSmallCopyBytes);
    pipe_.InitBuffer(tile_buf_, kSmallCopyBytes);
  }

  __aicore__ inline void Process(uint32_t repeats) {
    LocalTensor<uint8_t> tile = tile_buf_.Get<uint8_t>();
    for (uint32_t i = 0; i < repeats; ++i) {
      if ((i & 0x7ffu) == 0) {
        tile.SetValue(0, static_cast<uint8_t>(i));
      }
    }
    DataCopy(output_, tile, kSmallCopyBytes);
    PipeBarrier<PIPE_MTE3>();
  }

 private:
  TPipe pipe_;
  TBuf<QuePosition::VECCALC> tile_buf_;
  GlobalTensor<uint8_t> output_;
};
}  // namespace

extern "C" __global__ __aicore__ void mte_startup_latency_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  MteStartupKernel op;
  op.Init(input, output);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void mte_startup_latency_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  MteStartupBaselineKernel op;
  op.Init(output);
  op.Process(repeats);
}

#ifndef ASCENDC_CPU_DEBUG
void mte_startup_latency_target_do(uint32_t block_dim, aclrtStream stream,
                                   void* input, void* output,
                                   uint32_t size_bytes, uint32_t repeats,
                                   uint32_t mode) {
  mte_startup_latency_target_kernel<<<block_dim, nullptr, stream>>>(
      reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
      size_bytes, repeats, mode);
}

void mte_startup_latency_baseline_do(uint32_t block_dim, aclrtStream stream,
                                     void* input, void* output,
                                     uint32_t size_bytes, uint32_t repeats,
                                     uint32_t mode) {
  mte_startup_latency_baseline_kernel<<<block_dim, nullptr, stream>>>(
      reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
      size_bytes, repeats, mode);
}
#endif
