#include "bench_utils.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace ubench {
namespace {

uint32_t ParseU32(const char* value, const char* flag) {
  char* end = nullptr;
  const auto parsed = std::strtoul(value, &end, 0);
  if (end == value || *end != '\0') {
    throw std::runtime_error(std::string("Invalid value for ") + flag);
  }
  return static_cast<uint32_t>(parsed);
}

std::string RequireValue(int* index, int argc, char** argv) {
  if (*index + 1 >= argc) {
    throw std::runtime_error(std::string("Missing value for ") + argv[*index]);
  }
  ++(*index);
  return argv[*index];
}

}  // namespace

BenchOptions ParseArgs(int argc, char** argv) {
  BenchOptions opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--device") {
      opts.device = static_cast<int32_t>(
          ParseU32(RequireValue(&i, argc, argv).c_str(), "--device"));
    } else if (arg == "--warmup") {
      opts.warmup = ParseU32(RequireValue(&i, argc, argv).c_str(), "--warmup");
    } else if (arg == "--iters") {
      opts.iters = ParseU32(RequireValue(&i, argc, argv).c_str(), "--iters");
    } else if (arg == "--blocks") {
      opts.blocks = ParseU32(RequireValue(&i, argc, argv).c_str(), "--blocks");
    } else if (arg == "--repeats") {
      opts.repeats = ParseU32(RequireValue(&i, argc, argv).c_str(), "--repeats");
    } else if (arg == "--size") {
      opts.size_bytes = ParseU32(RequireValue(&i, argc, argv).c_str(), "--size");
    } else if (arg == "--csv") {
      opts.csv_path = RequireValue(&i, argc, argv);
    } else if (arg == "--profile") {
      opts.profile = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Options: --device N --warmup N --iters N --blocks N "
          << "--repeats N --size BYTES --csv PATH --profile\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  return opts;
}

std::vector<uint32_t> RepeatSweep(uint32_t base_repeats) {
  const uint32_t base = base_repeats == 0 ? 1 : base_repeats;
  return {base, base * 2, base * 5, base * 10};
}

FitResult FitSlope(const std::vector<TimingPoint>& points, bool subtract_baseline) {
  FitResult fit;
  if (points.size() < 2) {
    return fit;
  }

  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_xx = 0.0;
  double sum_xy = 0.0;
  for (const auto& p : points) {
    const double x = static_cast<double>(p.repeats);
    const double y = subtract_baseline ? (p.total_us - p.baseline_us) : p.total_us;
    sum_x += x;
    sum_y += y;
    sum_xx += x * x;
    sum_xy += x * y;
  }
  const double n = static_cast<double>(points.size());
  const double denom = n * sum_xx - sum_x * sum_x;
  if (std::abs(denom) < 1e-12) {
    return fit;
  }
  fit.slope_us = (n * sum_xy - sum_x * sum_y) / denom;
  fit.intercept_us = (sum_y - fit.slope_us * sum_x) / n;

  const double mean_y = sum_y / n;
  double ss_tot = 0.0;
  double ss_res = 0.0;
  for (const auto& p : points) {
    const double x = static_cast<double>(p.repeats);
    const double y = subtract_baseline ? (p.total_us - p.baseline_us) : p.total_us;
    const double pred = fit.intercept_us + fit.slope_us * x;
    ss_tot += (y - mean_y) * (y - mean_y);
    ss_res += (y - pred) * (y - pred);
  }
  fit.r2 = ss_tot <= 1e-12 ? 1.0 : 1.0 - ss_res / ss_tot;
  return fit;
}

void PrintSummary(const std::string& name,
                  const BenchOptions& opts,
                  const std::vector<TimingPoint>& points,
                  const FitResult& raw_fit,
                  const FitResult& adjusted_fit) {
  std::cout << "benchmark=" << name << "\n";
  std::cout << "device=" << opts.device << " blocks=" << opts.blocks
            << " size_bytes=" << opts.size_bytes << " warmup=" << opts.warmup
            << " iters=" << opts.iters << "\n";
  std::cout << "repeat,total_us,baseline_us,adjusted_us\n";
  for (const auto& p : points) {
    std::cout << p.repeats << "," << std::fixed << std::setprecision(3)
              << p.total_us << "," << p.baseline_us << ","
              << (p.total_us - p.baseline_us) << "\n";
  }
  std::cout << std::setprecision(9);
  std::cout << "raw_slope_us=" << raw_fit.slope_us
            << " raw_intercept_us=" << raw_fit.intercept_us
            << " raw_r2=" << raw_fit.r2 << "\n";
  std::cout << "adjusted_slope_us=" << adjusted_fit.slope_us
            << " adjusted_intercept_us=" << adjusted_fit.intercept_us
            << " adjusted_r2=" << adjusted_fit.r2 << "\n";
}

void WriteCsv(const std::string& path,
              const std::string& name,
              const BenchOptions& opts,
              const std::vector<TimingPoint>& points,
              const FitResult& raw_fit,
              const FitResult& adjusted_fit) {
  if (path.empty()) {
    return;
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not open CSV output: " + path);
  }
  out << "benchmark,device,blocks,size_bytes,repeat,total_us,baseline_us,"
      << "adjusted_us,raw_slope_us,raw_intercept_us,raw_r2,"
      << "adjusted_slope_us,adjusted_intercept_us,adjusted_r2\n";
  for (const auto& p : points) {
    out << name << "," << opts.device << "," << opts.blocks << ","
        << opts.size_bytes << "," << p.repeats << "," << p.total_us << ","
        << p.baseline_us << "," << (p.total_us - p.baseline_us) << ","
        << raw_fit.slope_us << "," << raw_fit.intercept_us << ","
        << raw_fit.r2 << "," << adjusted_fit.slope_us << ","
        << adjusted_fit.intercept_us << "," << adjusted_fit.r2 << "\n";
  }
}

void FillHostPattern(std::vector<uint8_t>* data) {
  for (size_t i = 0; i < data->size(); ++i) {
    (*data)[i] = static_cast<uint8_t>((i * 131u + 17u) & 0xffu);
  }
}

uint64_t Checksum(const std::vector<uint8_t>& data) {
  uint64_t sum = 0;
  for (uint8_t v : data) {
    sum = (sum * 1315423911ull) ^ static_cast<uint64_t>(v + 1u);
  }
  return sum;
}

}  // namespace ubench
