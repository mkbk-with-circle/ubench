#include "bench_runner.h"
#include "ubench_kernels.h"

int main(int argc, char** argv) {
  return ubench::RunBenchmark(
      "mte_granularity", argc, argv,
      {mte_granularity_target_do, mte_granularity_baseline_do}, 0);
}
