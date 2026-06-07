#include "bench_runner.h"
#include "ubench_kernels.h"

int main(int argc, char** argv) {
  return ubench::RunBenchmark(
      "cube_tile_latency", argc, argv,
      {cube_tile_latency_target_do, cube_tile_latency_baseline_do}, 0);
}
