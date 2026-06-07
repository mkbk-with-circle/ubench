#include "bench_runner.h"
#include "ubench_kernels.h"

int main(int argc, char** argv) {
  return ubench::RunBenchmark(
      "mte_copy_bw", argc, argv,
      {mte_copy_bw_target_do, mte_copy_bw_baseline_do}, 0);
}
