#include "bench_runner.h"
#include "ubench_kernels.h"

int main(int argc, char** argv) {
  return ubench::RunBenchmark(
      "vector_pipeline_depth", argc, argv,
      {vector_pipeline_depth_target_do, vector_pipeline_depth_baseline_do}, 0);
}
