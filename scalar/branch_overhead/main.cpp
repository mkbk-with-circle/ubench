#include "bench_runner.h"
#include "ubench_kernels.h"

int main(int argc, char** argv) {
  return ubench::RunBenchmark(
      "scalar_branch_overhead", argc, argv,
      {scalar_branch_overhead_target_do, scalar_branch_overhead_baseline_do}, 0);
}
