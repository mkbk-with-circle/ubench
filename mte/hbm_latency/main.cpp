#include "bench_runner.h"
#include "ubench_kernels.h"

int main(int argc, char** argv) {
  return ubench::RunBenchmark(
      "mte_hbm_latency", argc, argv,
      {mte_hbm_latency_target_do, mte_hbm_latency_baseline_do}, 0);
}
