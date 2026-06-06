#!/usr/bin/env python3
"""
Scalar Unit Microbenchmark Runner
Runs all Scalar Unit benchmarks (S1-S3) and saves results.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.benchmark import save_results, print_result

from .s1_latency import main as s1_main
from .s2_throughput import main as s2_main
from .s3_mem_latency import main as s3_main


def run_all():
    print("\n" + "=" * 70)
    print("  SCALAR UNIT MICROBENCHMARKS (S1-S3)")
    print("=" * 70)

    all_results = []

    print("\n>>> Running S1: Scalar Arithmetic Instruction Latency")
    results_s1 = s1_main()
    all_results.extend(results_s1)

    print("\n>>> Running S2: Scalar Unit Throughput")
    results_s2 = s2_main()
    all_results.extend(results_s2)

    print("\n>>> Running S3: Scalar Memory Access Latency")
    results_s3 = s3_main()
    all_results.extend(results_s3)

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    save_results(all_results, os.path.join(output_dir, "scalar_results.json"))

    print("\n" + "=" * 70)
    print("  SCALAR UNIT SUMMARY")
    print("=" * 70)
    for r in all_results:
        print(f"  [{r.param_id}] {r.param_name}: {r.value:.4f} {r.unit}")
        print(f"       (median={r.median:.4f}, cv={r.cv:.2%})")

    return all_results


if __name__ == "__main__":
    run_all()
