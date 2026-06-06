#!/usr/bin/env python3
"""
Ascend NPU Microbenchmark Suite — Master Runner
Runs all 21 microbenchmarks across 4 categories:
  - Scalar Unit (S1-S3)
  - Vector Unit (V1-V5)
  - Cube Unit (C1-C5)
  - MTE (M1-M8)

Usage:
  python run_all.py                    # Run all benchmarks
  python run_all.py --category scalar  # Run only scalar benchmarks
  python run_all.py --category vector  # Run only vector benchmarks
  python run_all.py --category cube    # Run only cube benchmarks
  python run_all.py --category mte     # Run only MTE benchmarks
  python run_all.py --quick            # Quick mode (fewer iterations)
"""

import sys
import os
import argparse
import time

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

from common.benchmark import save_results


def run_category(category: str, quick: bool = False):
    """Run all benchmarks in a given category."""
    results = []

    if category == "scalar":
        from scalar import run_all
        results = run_all()
    elif category == "vector":
        from vector import run_all
        results = run_all()
    elif category == "cube":
        from cube import run_all
        results = run_all()
    elif category == "mte":
        from mte import run_all
        results = run_all()
    else:
        print(f"[ERROR] Unknown category: {category}")
        return []

    return results


def run_all_categories(quick: bool = False):
    """Run all benchmark categories."""
    all_results = []

    categories = ["scalar", "vector", "cube", "mte"]
    for cat in categories:
        print(f"\n{'#' * 70}")
        print(f"  Running {cat.upper()} benchmarks...")
        print(f"{'#' * 70}")
        results = run_category(cat, quick)
        all_results.extend(results)

    return all_results


def print_summary(all_results):
    """Print a summary of all benchmark results."""
    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK SUITE SUMMARY")
    print(f"  Total measurements: {len(all_results)}")
    print(f"{'=' * 70}")

    # Group by category
    from collections import defaultdict
    by_category = defaultdict(list)
    for r in all_results:
        by_category[r.category].append(r)

    for cat, results in sorted(by_category.items()):
        print(f"\n  [{cat}]")
        for r in results:
            print(f"    {r.param_id:8s} {r.param_name:40s} = {r.value:10.4f} {r.unit}")


def main():
    parser = argparse.ArgumentParser(
        description="Ascend NPU Microbenchmark Suite"
    )
    parser.add_argument(
        "--category", "-c",
        choices=["scalar", "vector", "cube", "mte", "all"],
        default="all",
        help="Which category to benchmark (default: all)"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: fewer warmup/iteration rounds"
    )
    parser.add_argument(
        "--output", "-o",
        default="results/all_results.json",
        help="Output file for aggregated results"
    )

    args = parser.parse_args()

    start_time = time.time()

    if args.category == "all":
        all_results = run_all_categories(quick=args.quick)
    else:
        all_results = run_category(args.category, quick=args.quick)

    elapsed = time.time() - start_time

    # Save aggregated results
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    save_results(all_results, output_path)

    # Print summary
    print_summary(all_results)

    print(f"\n{'=' * 70}")
    print(f"  Total time: {elapsed:.1f} seconds")
    print(f"  Results saved to: {output_path}")
    print(f"{'=' * 70}")

    return all_results


if __name__ == "__main__":
    main()
