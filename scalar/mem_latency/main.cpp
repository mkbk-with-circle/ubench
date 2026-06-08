#include "bench_runner.h"
#include "ubench_kernels.h"

int main(int argc, char** argv) {
  return ubench::RunBenchmark(
      "scalar_mem_latency", argc, argv,
      {scalar_mem_latency_target_do, scalar_mem_latency_baseline_do}, 0);
}
