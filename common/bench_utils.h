#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace ubench {

struct BenchOptions {
  int32_t device = 0;
  uint32_t warmup = 5;
  uint32_t iters = 20;
  uint32_t blocks = 8;
  uint32_t repeats = 1000;
  uint32_t size_bytes = 1 << 20;
  bool profile = false;
  std::string csv_path;
};

struct TimingPoint {
  uint32_t repeats = 0;
  double total_us = 0.0;
  double baseline_us = 0.0;
};

struct FitResult {
  double intercept_us = 0.0;
  double slope_us = 0.0;
  double r2 = 0.0;
};

BenchOptions ParseArgs(int argc, char** argv);
std::vector<uint32_t> RepeatSweep(uint32_t base_repeats);
FitResult FitSlope(const std::vector<TimingPoint>& points, bool subtract_baseline);
void PrintSummary(const std::string& name,
                  const BenchOptions& opts,
                  const std::vector<TimingPoint>& points,
                  const FitResult& raw_fit,
                  const FitResult& adjusted_fit);
void WriteCsv(const std::string& path,
              const std::string& name,
              const BenchOptions& opts,
              const std::vector<TimingPoint>& points,
              const FitResult& raw_fit,
              const FitResult& adjusted_fit);
void FillHostPattern(std::vector<uint8_t>* data);
uint64_t Checksum(const std::vector<uint8_t>& data);

}  // namespace ubench
