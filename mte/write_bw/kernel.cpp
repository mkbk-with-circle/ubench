#include "kernel_operator.h"

#ifndef ASCENDC_CPU_DEBUG
#include <acl/acl.h>
#endif

using namespace AscendC;

namespace {
// M2: UB→GM write-only bandwidth
constexpr uint32_t kTileBytes = 1024;

class MteWriteBwKernel {
 public:
  __aicore__ inline void Init(GM_ADDR output, uint32_t size_bytes) {
    const uint32_t block = GetBlockIdx();
    const uint32_t block_offset = block * kTileBytes;
    const uint32_t safe_offset = block_offset < size_bytes ? block_offset : 0;
    output_.SetGlobalBuffer((__gm__ uint8_t*)output + safe_offset, kTileBytes);
    pipe_.InitBuffer(tile_buf_, kTileBytes);
  }

  __aicore__ inline void Process(uint32_t repeats) {
    LocalTensor<uint8_t> tile = tile_buf_.Get<uint8_t>();
    // Fill tile with pattern
    for (uint32_t i = 0; i < kTileBytes; ++i) {
      tile.SetValue(i, static_cast<uint8_t>(i & 0xff));
    }
    for (uint32_t i = 0; i < repeats; ++i) {
      DataCopy(output_, tile, kTileBytes);
      PipeBarrier<PIPE_MTE3>();
    }
  }

 private:
  TPipe pipe_;
  TBuf<QuePosition::VECCALC> tile_buf_;
  GlobalTensor<uint8_t> output_;
};

class MteWriteBwBaselineKernel {
 public:
  __aicore__ inline void Init(GM_ADDR output, uint32_t size_bytes) {
    const uint32_t safe_offset = GetBlockIdx() < size_bytes ? GetBlockIdx() : 0;
    output_.SetGlobalBuffer((__gm__ uint8_t*)output + safe_offset, 32);
    pipe_.InitBuffer(tile_buf_, 32);
  }

  __aicore__ inline void Process(uint32_t repeats) {
    LocalTensor<uint8_t> tile = tile_buf_.Get<uint8_t>();
    for (uint32_t i = 0; i < repeats; ++i) {
      if ((i & 0x3ffu) == 0) {
        DataCopy(output_, tile, 32);
        PipeBarrier<PIPE_MTE3>();
      }
    }
  }

 private:
  TPipe pipe_;
  TBuf<QuePosition::VECCALC> tile_buf_;
  GlobalTensor<uint8_t> output_;
};
}  // namespace

extern "C" __global__ __aicore__ void mte_write_bw_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  MteWriteBwKernel op;
  op.Init(output, size_bytes);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void mte_write_bw_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  MteWriteBwBaselineKernel op;
  op.Init(output, size_bytes);
  op.Process(repeats);
}

#ifndef ASCENDC_CPU_DEBUG
void mte_write_bw_target_do(uint32_t block_dim, aclrtStream stream, void* input,
                             void* output, uint32_t size_bytes, uint32_t repeats,
                             uint32_t mode) {
  mte_write_bw_target_kernel<<<block_dim, nullptr, stream>>>(
      reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
      size_bytes, repeats, mode);
}

void mte_write_bw_baseline_do(uint32_t block_dim, aclrtStream stream, void* input,
                               void* output, uint32_t size_bytes, uint32_t repeats,
                               uint32_t mode) {
  mte_write_bw_baseline_kernel<<<block_dim, nullptr, stream>>>(
      reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
      size_bytes, repeats, mode);
}
#endif
