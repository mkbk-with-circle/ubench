// C5: Cube scaling benchmark -- sweeps matrix dimensions via mode parameter
// mode 0 → 16×16×16, mode 1 → 32×32×32, mode 2 → 64×64×64, mode 3 → 128×128×128
#include "acl_utils.h"
#include "bench_runner.h"
#include "bench_utils.h"
#include "ubench_kernels.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static constexpr uint32_t kMaxInputBytes = 128 * 128 * 2 * 2;  // 128×128 FP16 × 2 matrices
static constexpr uint32_t kMaxOutputBytes = 128 * 128 * 4;     // 128×128 FP32

int main(int argc, char** argv) {
  try {
    ubench::BenchOptions opts = ubench::ParseArgs(argc, argv);
    ubench::AclRuntime runtime(opts.device);

    // Allocate buffers large enough for the biggest matrix (128×128)
    const size_t in_bytes = kMaxInputBytes;
    const size_t out_bytes = kMaxOutputBytes;

    std::vector<uint8_t> input_host(in_bytes);
    ubench::FillHostPattern(&input_host);

    ubench::DeviceBuffer input_dev(in_bytes);
    ubench::DeviceBuffer output_dev(out_bytes);
    CHECK_ACL(aclrtMemcpy(input_dev.get(), in_bytes, input_host.data(), in_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemset(output_dev.get(), out_bytes, 0, out_bytes));

    // Warmup with mode=0 (16×16)
    for (uint32_t i = 0; i < opts.warmup; ++i) {
      cube_scaling_target_do(opts.blocks, runtime.stream(), input_dev.get(),
                             output_dev.get(), in_bytes, opts.repeats, 0);
    }
    CHECK_ACL(aclrtSynchronizeStream(runtime.stream()));

    // Open CSV for combined output
    std::ofstream csv;
    if (!opts.csv_path.empty()) {
      csv.open(opts.csv_path);
      csv << "benchmark,device,blocks,size_bytes,repeat,mode,dim,"
          << "total_us,baseline_us,adjusted_us,"
          << "raw_slope_us,raw_intercept_us,raw_r2,"
          << "adjusted_slope_us,adjusted_intercept_us,adjusted_r2\n";
    }

    // Sweep modes 0-3 (16, 32, 64, 128)
    for (uint32_t mode = 0; mode <= 3; ++mode) {
      const uint32_t dim = (mode == 0) ? 16 : (mode == 1) ? 32 : (mode == 2) ? 64 : 128;
      std::cout << "\n=== cube_scaling mode=" << mode << " dim=" << dim << " ===\n";

      std::vector<ubench::TimingPoint> points;
      for (uint32_t repeats : ubench::RepeatSweep(opts.repeats)) {
        ubench::TimingPoint point;
        point.repeats = repeats;

        // Target timing
        CHECK_ACL(aclrtSynchronizeStream(runtime.stream()));
        double begin = ubench::NowMicros();
        for (uint32_t i = 0; i < opts.iters; ++i) {
          cube_scaling_target_do(opts.blocks, runtime.stream(), input_dev.get(),
                                 output_dev.get(), in_bytes, repeats, mode);
        }
        CHECK_ACL(aclrtSynchronizeStream(runtime.stream()));
        double end = ubench::NowMicros();
        point.total_us = (end - begin) / static_cast<double>(opts.iters);

        // Baseline timing
        CHECK_ACL(aclrtSynchronizeStream(runtime.stream()));
        begin = ubench::NowMicros();
        for (uint32_t i = 0; i < opts.iters; ++i) {
          cube_scaling_baseline_do(opts.blocks, runtime.stream(), input_dev.get(),
                                   output_dev.get(), in_bytes, repeats, mode);
        }
        CHECK_ACL(aclrtSynchronizeStream(runtime.stream()));
        end = ubench::NowMicros();
        point.baseline_us = (end - begin) / static_cast<double>(opts.iters);

        points.push_back(point);
      }

      const auto raw_fit = ubench::FitSlope(points, false);
      const auto adj_fit = ubench::FitSlope(points, true);

      std::cout << "raw_slope_us=" << raw_fit.slope_us
                << " raw_r2=" << raw_fit.r2 << "\n";
      std::cout << "adjusted_slope_us=" << adj_fit.slope_us
                << " adjusted_r2=" << adj_fit.r2 << "\n";

      if (csv.is_open()) {
        for (const auto& p : points) {
          csv << "cube_scaling," << opts.device << "," << opts.blocks << ","
              << in_bytes << "," << p.repeats << "," << mode << "," << dim << ","
              << p.total_us << "," << p.baseline_us << ","
              << (p.total_us - p.baseline_us) << ","
              << raw_fit.slope_us << "," << raw_fit.intercept_us << ","
              << raw_fit.r2 << "," << adj_fit.slope_us << ","
              << adj_fit.intercept_us << "," << adj_fit.r2 << "\n";
        }
      }
    }

    std::cout << "\n[cube_scaling] done -- results in " << opts.csv_path << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
