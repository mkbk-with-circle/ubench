#include "kernel_operator.h"

#ifndef ASCENDC_CPU_DEBUG
#include <acl/acl.h>
#endif

using namespace AscendC;

namespace {
// M7: Buffer capacity curve — sweep working-set size
// mode selects the copy size: 32B, 64B, 128B, 256B, 512B, 1KB, 2KB, 4KB, 8KB, 16KB, 32KB, 64KB
// Each mode value maps to a size via: size = 32 << mode (clamped to size_bytes)

__aicore__ inline uint32_t ModeToBytes(uint32_t mode, uint32_t size_bytes) {
  uint32_t bytes = 32u << (mode > 11u ? 11u : mode);
  return bytes > size_bytes ? size_bytes : bytes;
}

class MteBufferCapacityKernel {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t mode) {
    copy_bytes_ = ModeToBytes(mode, size_bytes);
    if (copy_bytes_ < 32) copy_bytes_ = 32;
    const uint32_t block = GetBlockIdx();
    const uint32_t offset = (block * copy_bytes_) % (size_bytes > copy_bytes_ ? size_bytes - copy_bytes_ : 1);
    input_ptr_ = (__gm__ uint8_t*)input + offset;
    output_ptr_ = (__gm__ uint8_t*)output + offset;
    // Clamp tile buffer to 1024B max for L1 constraints
    tile_size_ = copy_bytes_ > 1024 ? 1024 : copy_bytes_;
    pipe_.InitBuffer(tile_buf_, tile_size_);
  }

  __aicore__ inline void Process(uint32_t repeats) {
    LocalTensor<uint8_t> tile = tile_buf_.Get<uint8_t>();
    for (uint32_t i = 0; i < repeats; ++i) {
      GlobalTensor<uint8_t> in_slice, out_slice;
      in_slice.SetGlobalBuffer(input_ptr_, copy_bytes_);
      out_slice.SetGlobalBuffer(output_ptr_, copy_bytes_);
      DataCopy(tile, in_slice, copy_bytes_);
      PipeBarrier<PIPE_MTE2>();
      DataCopy(out_slice, tile, copy_bytes_);
      PipeBarrier<PIPE_MTE3>();
    }
  }

 private:
  uint32_t copy_bytes_ = 32;
  uint32_t tile_size_ = 32;
  __gm__ uint8_t* input_ptr_;
  __gm__ uint8_t* output_ptr_;
  TPipe pipe_;
  TBuf<QuePosition::VECCALC> tile_buf_;
};

class MteBufferCapacityBaselineKernel {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t size_bytes) {
    const uint32_t safe_offset = GetBlockIdx() < size_bytes ? GetBlockIdx() : 0;
    input_.SetGlobalBuffer((__gm__ uint8_t*)input + safe_offset, 32);
    output_.SetGlobalBuffer((__gm__ uint8_t*)output + safe_offset, 32);
    pipe_.InitBuffer(tile_buf_, 32);
  }

  __aicore__ inline void Process(uint32_t repeats) {
    LocalTensor<uint8_t> tile = tile_buf_.Get<uint8_t>();
    for (uint32_t i = 0; i < repeats; ++i) {
      if ((i & 0x3ffu) == 0) {
        DataCopy(tile, input_, 32);
        PipeBarrier<PIPE_MTE2>();
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

extern "C" __global__ __aicore__ void mte_buffer_capacity_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  MteBufferCapacityKernel op;
  op.Init(input, output, size_bytes, mode);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void mte_buffer_capacity_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats, uint32_t mode) {
  MteBufferCapacityBaselineKernel op;
  op.Init(input, output, size_bytes);
  op.Process(repeats);
}

#ifndef ASCENDC_CPU_DEBUG
void mte_buffer_capacity_target_do(uint32_t block_dim, aclrtStream stream, void* input,
                                    void* output, uint32_t size_bytes, uint32_t repeats,
                                    uint32_t mode) {
  mte_buffer_capacity_target_kernel<<<block_dim, nullptr, stream>>>(
      reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
      size_bytes, repeats, mode);
}

void mte_buffer_capacity_baseline_do(uint32_t block_dim, aclrtStream stream, void* input,
                                      void* output, uint32_t size_bytes, uint32_t repeats,
                                      uint32_t mode) {
  mte_buffer_capacity_baseline_kernel<<<block_dim, nullptr, stream>>>(
      reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
      size_bytes, repeats, mode);
}
#endif
