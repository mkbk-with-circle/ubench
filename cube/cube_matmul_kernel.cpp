#include "kernel_operator.h"
#include "matmul_intf.h"

#ifndef ASCENDC_CPU_DEBUG
#include <acl/acl.h>
#endif

using namespace AscendC;

namespace {
// Minimal Cube test: DataCopy through L1 to verify AIC binary works
// If this works, we can add Matmul later
constexpr uint32_t kTileBytes = 256;

class CubeTestKernel {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t size_bytes) {
    const uint32_t block = GetBlockIdx();
    const uint32_t offset = (block * kTileBytes) % (size_bytes > kTileBytes ? size_bytes - kTileBytes : 1);
    input_.SetGlobalBuffer((__gm__ uint8_t*)input + offset, kTileBytes);
    output_.SetGlobalBuffer((__gm__ uint8_t*)output + offset, kTileBytes);
    pipe_.InitBuffer(tile_buf_, kTileBytes);
  }

  __aicore__ inline void Process(uint32_t repeats) {
    LocalTensor<uint8_t> tile = tile_buf_.Get<uint8_t>();
    for (uint32_t i = 0; i < repeats; ++i) {
      DataCopy(tile, input_, kTileBytes);
      PipeBarrier<PIPE_MTE2>();
      DataCopy(output_, tile, kTileBytes);
      PipeBarrier<PIPE_MTE3>();
    }
  }

 private:
  TPipe pipe_;
  TBuf<QuePosition::VECCALC> tile_buf_;
  GlobalTensor<uint8_t> input_;
  GlobalTensor<uint8_t> output_;
};

class CubeBaselineKernel {
 public:
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
    const uint32_t offset = GetBlockIdx() * 32;
    input_.SetGlobalBuffer((__gm__ uint8_t*)input + offset, 32);
    output_.SetGlobalBuffer((__gm__ uint8_t*)output + offset, 32);
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

extern "C" __global__ __aicore__ void cube_tile_latency_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats,
    uint32_t mode) {
  CubeTestKernel op;
  op.Init(input, output, size_bytes);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void cube_tile_latency_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats,
    uint32_t mode) {
  CubeBaselineKernel op;
  op.Init(input, output);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void cube_throughput_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats,
    uint32_t mode) {
  CubeTestKernel op;
  op.Init(input, output, size_bytes);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void cube_throughput_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats,
    uint32_t mode) {
  CubeBaselineKernel op;
  op.Init(input, output);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void cube_scaling_target_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats,
    uint32_t mode) {
  CubeTestKernel op;
  op.Init(input, output, size_bytes);
  op.Process(repeats);
}

extern "C" __global__ __aicore__ void cube_scaling_baseline_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t size_bytes, uint32_t repeats,
    uint32_t mode) {
  CubeBaselineKernel op;
  op.Init(input, output);
  op.Process(repeats);
}

#ifndef ASCENDC_CPU_DEBUG
#define DEFINE_CUBE_DO(kernel_name)                                            \
  void kernel_name##_do(uint32_t block_dim, aclrtStream stream, void* input,   \
                        void* output, uint32_t size_bytes, uint32_t repeats,   \
                        uint32_t mode) {                                       \
    kernel_name##_kernel<<<block_dim, nullptr, stream>>>(                      \
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
