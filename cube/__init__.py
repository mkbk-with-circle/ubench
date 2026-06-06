#!/usr/bin/env python3
"""
Cube Unit Microbenchmark Runner
Runs all Cube Unit benchmarks (C1-C5) and saves results.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.benchmark import save_results

from .c1_tile_latency import main as c1_main
from .c2_throughput import main as c2_main
from .c3_pipeline_depth import main as c3_main
from .c4_l0_latency import main as c4_main
from .c5_scaling import main as c5_main


def run_all():
    print("\n" + "=" * 70)
    print("  CUBE UNIT MICROBENCHMARKS (C1-C5)")
    print("=" * 70)

    all_results = []

    print("\n>>> Running C1: Single Tile MatMul Latency")
    results_c1 = c1_main()
    all_results.extend(results_c1)

    print("\n>>> Running C2: MatMul Throughput")
    results_c2 = c2_main()
    all_results.extend(results_c2)

    print("\n>>> Running C3: Cube Pipeline Depth")
    results_c3 = c3_main()
    all_results.extend(results_c3)

    print("\n>>> Running C4: L0A/L0B/L0C Buffer Latency")
    results_c4 = c4_main()
    all_results.extend(results_c4)

    print("\n>>> Running C5: MatMul Latency Scaling")
    results_c5 = c5_main()
    all_results.extend(results_c5)

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    save_results(all_results, os.path.join(output_dir, "cube_results.json"))

    print("\n" + "=" * 70)
    print("  CUBE UNIT SUMMARY")
    print("=" * 70)
    for r in all_results:
        print(f"  [{r.param_id}] {r.param_name}: {r.value:.4f} {r.unit}")

    return all_results


if __name__ == "__main__":
    run_all()
