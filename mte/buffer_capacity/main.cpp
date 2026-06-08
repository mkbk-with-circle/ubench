#include "bench_runner.h"
#include "ubench_kernels.h"

int main(int argc, char** argv) {
  return ubench::RunBenchmark(
      "mte_buffer_capacity", argc, argv,
      {mte_buffer_capacity_target_do, mte_buffer_capacity_baseline_do}, 0);
}
