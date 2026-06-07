#include "bench_runner.h"
#include "ubench_kernels.h"

int main(int argc, char** argv) {
  return ubench::RunBenchmark(
      "cube_throughput", argc, argv,
      {cube_throughput_target_do, cube_throughput_baseline_do}, 0);
}
