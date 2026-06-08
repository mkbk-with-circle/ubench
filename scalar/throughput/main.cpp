#include "bench_runner.h"
#include "ubench_kernels.h"

int main(int argc, char** argv) {
  return ubench::RunBenchmark(
      "scalar_throughput", argc, argv,
      {scalar_throughput_target_do, scalar_throughput_baseline_do}, 0);
}
