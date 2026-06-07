#include "bench_runner.h"
#include "ubench_kernels.h"

int main(int argc, char** argv) {
  return ubench::RunBenchmark(
      "vector_mul_latency", argc, argv,
      {vector_mul_latency_target_do, vector_mul_latency_baseline_do}, 0);
}
