#include "kernel_operator.h"

#ifndef ASCENDC_CPU_DEBUG
#include <acl/acl.h>
#endif

using namespace AscendC;

namespace {
// M6: HBM access latency via dependent DataCopy chain
// Each copy's completion blocks the next → measures per-copy latency
constexpr uint32_t kCopyBytes = 32;  // Minimum granularity

class MteHbmLatencyKernel {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t size_bytes) {
    const uint32_t block = GetBlockIdx();
    const uint32_t offset = (block * kCopyBytes) % (size_bytes > kCopyBytes ? size_bytes - kCopyBytes : 1);
    input_.SetGlobalBuffer((__gm__ uint8_t*)input + offset, kCopyBytes);
    output_.SetGlobalBuffer((__gm__ uint8_t*)output + offset, kCopyBytes);
    pipe_.InitBuffer(tile_buf_, kCopyBytes);
  }

  __aicore__ inline void Process(uint32_t repeats) {
    LocalTensor<uint8_t> tile = tile_buf_.Get<uint8_t>();
    for (uint32_t i = 0; i < repeats; ++i) {
      // Dependent chain: read then write, with barrier between each
      DataCopy(tile, input_, kCopyBytes);
      PipeBarrier<PIPE_MTE2>();
      DataCopy(output_, tile, kCopyBytes);
      PipeBarrier<PIPE_MTE3>();
    }
  }

 private:
  TPipe pipe_;
  TBuf<QuePosition::VECCALC> tile_buf_;
  GlobalTensor<uint8_t> input_;
  GlobalTensor<uint8_t> output_;
};

class MteHbmLatencyBaselineKernel {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t size_bytes) {
    const uint32_t offset = GetBlockIdx() < size_bytes ? GetBlockIdx() : 0;
    input_.SetGlobalBuffer((__gm__ uint8_t*)input + offset, 32);
    output_.SetGlobalBuffer((__gm__ uint8_t*)output + offset, 32);
    pipe_.InitBuffer(tile_buf_, 32);
  }

  __aicore__ inline void Process(uint32_t repeats) {
    LocalTensor<uint8_t> tile = tile_buf_.Get<uint8_t>();
    for (uint32_t i = 0; i < repeats; ++i) {
      if ((i & 0x7ffu) == 0) {
        tile.SetValue(0, static_cast<uint8_t>(i));
      }
    }
    DataCopy(output_, tile, 32);
    PipeBarrier<PIPE_MTE3>();
  }

 private:
  TPipe pipe_;
  TBuf<QuePosition::VECCALC> tile_buf_;
  GlobalTensor<uint8_t> input_;
  GlobalTensor<uint8_t> output_;
};
}  // namespace

extern "C" __global__ __aicore__ void mte_hbm_latency_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  MteHbmLatencyKernel op;
  op.Init(input, output, size_bytes);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void mte_hbm_latency_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  MteHbmLatencyBaselineKernel op;
  op.Init(input, output, size_bytes);
  op.Process(repeats);
}

#ifndef ASCENDC_CPU_DEBUG
void mte_hbm_latency_target_do(uint32_t block_dim, aclrtStream stream, void* input,
                                void* output, uint32_t size_bytes, uint32_t repeats,
                                uint32_t mode) {
  mte_hbm_latency_target_kernel<<<block_dim, nullptr, stream>>>(
      reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
      size_bytes, repeats, mode);
}

void mte_hbm_latency_baseline_do(uint32_t block_dim, aclrtStream stream, void* input,
                                  void* output, uint32_t size_bytes, uint32_t repeats,
                                  uint32_t mode) {
  mte_hbm_latency_baseline_kernel<<<block_dim, nullptr, stream>>>(
      reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
      size_bytes, repeats, mode);
}
#endif
