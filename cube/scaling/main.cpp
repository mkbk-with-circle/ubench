#include "bench_runner.h"
#include "ubench_kernels.h"

int main(int argc, char** argv) {
  return ubench::RunBenchmark(
      "cube_scaling", argc, argv,
      {cube_scaling_target_do, cube_scaling_baseline_do}, 0);
}
