#pragma once

#include "acl_utils.h"
#include "bench_utils.h"

#include <cstdint>
#include <iostream>
#include <vector>

namespace ubench {

using LaunchFn = void (*)(uint32_t block_dim,
                          aclrtStream stream,
                          void* input,
                          void* output,
                          uint32_t size_bytes,
                          uint32_t repeats,
                          uint32_t mode);

struct KernelPair {
  LaunchFn target = nullptr;
  LaunchFn baseline = nullptr;
};

inline double LaunchAndTime(KernelPair pair,
                            bool baseline,
                            aclrtStream stream,
                            void* input,
                            void* output,
                            uint32_t size_bytes,
                            uint32_t repeats,
                            uint32_t blocks,
                            uint32_t mode,
                            uint32_t iters) {
  const LaunchFn fn = baseline ? pair.baseline : pair.target;
  if (fn == nullptr) {
    return 0.0;
  }
  CHECK_ACL(aclrtSynchronizeStream(stream));
  const double begin = NowMicros();
  for (uint32_t i = 0; i < iters; ++i) {
    fn(blocks, stream, input, output, size_bytes, repeats, mode);
  }
  CHECK_ACL(aclrtSynchronizeStream(stream));
  const double end = NowMicros();
  return (end - begin) / static_cast<double>(iters);
}

inline int RunBenchmark(const char* name,
                        int argc,
                        char** argv,
                        KernelPair kernels,
                        uint32_t mode = 0) {
  try {
    BenchOptions opts = ParseArgs(argc, argv);
    AclRuntime runtime(opts.device);

    const size_t bytes = opts.size_bytes == 0 ? 4096 : opts.size_bytes;
    std::vector<uint8_t> input_host(bytes);
    std::vector<uint8_t> output_host(bytes);
    FillHostPattern(&input_host);

    DeviceBuffer input_dev(bytes);
    DeviceBuffer output_dev(bytes);
    CHECK_ACL(aclrtMemcpy(input_dev.get(), bytes, input_host.data(), bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemset(output_dev.get(), bytes, 0, bytes));

    for (uint32_t i = 0; i < opts.warmup; ++i) {
      kernels.target(opts.blocks, runtime.stream(), input_dev.get(), output_dev.get(),
                     static_cast<uint32_t>(bytes), opts.repeats, mode);
    }
    CHECK_ACL(aclrtSynchronizeStream(runtime.stream()));

    std::vector<TimingPoint> points;
    for (uint32_t repeats : RepeatSweep(opts.repeats)) {
      TimingPoint point;
      point.repeats = repeats;
      point.total_us =
          LaunchAndTime(kernels, false, runtime.stream(), input_dev.get(),
                        output_dev.get(), static_cast<uint32_t>(bytes), repeats,
                        opts.blocks, mode, opts.iters);
      point.baseline_us =
          LaunchAndTime(kernels, true, runtime.stream(), input_dev.get(),
                        output_dev.get(), static_cast<uint32_t>(bytes), repeats,
                        opts.blocks, mode, opts.iters);
      points.push_back(point);
    }

    CHECK_ACL(aclrtMemcpy(output_host.data(), bytes, output_dev.get(), bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    const uint64_t checksum = Checksum(output_host);
    const FitResult raw_fit = FitSlope(points, false);
    const FitResult adjusted_fit = FitSlope(points, true);
    PrintSummary(name, opts, points, raw_fit, adjusted_fit);
    std::cout << "output_checksum=" << checksum << "\n";
    WriteCsv(opts.csv_path, name, opts, points, raw_fit, adjusted_fit);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

}  // namespace ubench
