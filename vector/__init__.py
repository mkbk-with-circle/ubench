#!/usr/bin/env python3
"""
Vector Unit Microbenchmark Runner
Runs all Vector Unit benchmarks (V1-V5) and saves results.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.benchmark import save_results

from .v1_add_latency import main as v1_main
from .v2_mul_latency import main as v2_main
from .v3_throughput import main as v3_main
from .v4_pipeline_depth import main as v4_main
from .v5_reg_latency import main as v5_main


def run_all():
    print("\n" + "=" * 70)
    print("  VECTOR UNIT MICROBENCHMARKS (V1-V5)")
    print("=" * 70)

    all_results = []

    print("\n>>> Running V1: FP32 Vector Addition Latency")
    results_v1 = v1_main()
    all_results.extend(results_v1)

    print("\n>>> Running V2: FP32 Vector Multiplication Latency")
    results_v2 = v2_main()
    all_results.extend(results_v2)

    print("\n>>> Running V3: Vector Unit Throughput")
    results_v3 = v3_main()
    all_results.extend(results_v3)

    print("\n>>> Running V4: Vector Unit Pipeline Depth")
    results_v4 = v4_main()
    all_results.extend(results_v4)

    print("\n>>> Running V5: Vector Register Access Latency")
    results_v5 = v5_main()
    all_results.extend(results_v5)

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    save_results(all_results, os.path.join(output_dir, "vector_results.json"))

    print("\n" + "=" * 70)
    print("  VECTOR UNIT SUMMARY")
    print("=" * 70)
    for r in all_results:
        print(f"  [{r.param_id}] {r.param_name}: {r.value:.4f} {r.unit}")

    return all_results


if __name__ == "__main__":
    run_all()
