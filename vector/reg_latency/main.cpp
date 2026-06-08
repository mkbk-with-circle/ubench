#include "bench_runner.h"
#include "ubench_kernels.h"

int main(int argc, char** argv) {
  return ubench::RunBenchmark(
      "vector_reg_latency", argc, argv,
      {vector_reg_latency_target_do, vector_reg_latency_baseline_do}, 0);
}
