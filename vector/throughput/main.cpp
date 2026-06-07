#include "bench_runner.h"
#include "ubench_kernels.h"

int main(int argc, char** argv) {
  return ubench::RunBenchmark(
      "vector_throughput", argc, argv,
      {vector_throughput_target_do, vector_throughput_baseline_do}, 0);
}
