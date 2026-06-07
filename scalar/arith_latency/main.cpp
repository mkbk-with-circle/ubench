#include "bench_runner.h"
#include "ubench_kernels.h"

int main(int argc, char** argv) {
  return ubench::RunBenchmark(
      "scalar_arith_latency", argc, argv,
      {scalar_arith_latency_target_do, scalar_arith_latency_baseline_do}, 0);
}
