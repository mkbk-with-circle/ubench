#include "kernel_operator.h"

#ifndef ASCENDC_CPU_DEBUG
#include <acl/acl.h>
#endif

using namespace AscendC;

namespace {
constexpr uint32_t kMaxTileBytes = 512;

__aicore__ inline uint32_t ClampCopyBytes(uint32_t mode) {
  // mode controls the requested copy granularity in 32B units.
  const uint32_t units = mode == 0 ? 1 : mode;
  const uint32_t bytes = units * 32;
  return bytes > kMaxTileBytes ? kMaxTileBytes : bytes;
}

class MteGranularityKernel {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t size_bytes,
                              uint32_t mode) {
    copy_bytes_ = ClampCopyBytes(mode);
    const uint32_t block = GetBlockIdx();
    const uint32_t offset = (block * kMaxTileBytes) % (size_bytes - kMaxTileBytes + 1);
    input_.SetGlobalBuffer((__gm__ uint8_t*)input + offset, kMaxTileBytes);
    output_.SetGlobalBuffer((__gm__ uint8_t*)output + offset, kMaxTileBytes);
    pipe_.InitBuffer(tile_buf_, kMaxTileBytes);
  }

  __aicore__ inline void Process(uint32_t repeats) {
    LocalTensor<uint8_t> tile = tile_buf_.Get<uint8_t>();
    for (uint32_t i = 0; i < repeats; ++i) {
      DataCopy(tile, input_, copy_bytes_);
      PipeBarrier<PIPE_MTE2>();
      DataCopy(output_, tile, copy_bytes_);
      PipeBarrier<PIPE_MTE3>();
    }
  }

 private:
  uint32_t copy_bytes_ = 32;
  TPipe pipe_;
  TBuf<QuePosition::VECCALC> tile_buf_;
  GlobalTensor<uint8_t> input_;
  GlobalTensor<uint8_t> output_;
};
}  // namespace

extern "C" __global__ __aicore__ void mte_granularity_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats,
    uint32_t mode) {
  MteGranularityKernel op;
  op.Init(input, output, size_bytes < kMaxTileBytes ? kMaxTileBytes : size_bytes,
          mode);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void mte_granularity_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats,
    uint32_t mode) {
  MteGranularityKernel op;
  op.Init(input, output, size_bytes < kMaxTileBytes ? kMaxTileBytes : size_bytes,
          1);
  op.Process(repeats == 0 ? 0 : 1);
}

#ifndef ASCENDC_CPU_DEBUG
void mte_granularity_target_do(uint32_t block_dim, aclrtStream stream,
                               void* input, void* output, uint32_t size_bytes,
                               uint32_t repeats, uint32_t mode) {
  mte_granularity_target_kernel<<<block_dim, nullptr, stream>>>(
      reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
      size_bytes, repeats, mode);
}

void mte_granularity_baseline_do(uint32_t block_dim, aclrtStream stream,
                                 void* input, void* output, uint32_t size_bytes,
                                 uint32_t repeats, uint32_t mode) {
  mte_granularity_baseline_kernel<<<block_dim, nullptr, stream>>>(
      reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
      size_bytes, repeats, mode);
}
#endif
