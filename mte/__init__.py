#!/usr/bin/env python3
"""
MTE (Memory Transfer Engine) Microbenchmark Runner
Runs all MTE benchmarks (M1-M8) and saves results.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.benchmark import save_results

from .m1_l1_read_bw import main as m1_main
from .m2_l1_write_bw import main as m2_main
from .m3_m4_m5_l0_bw import main as m3_main
from .m6_hbm_latency import main as m6_main
from .m7_buffer_capacity import main as m7_main
from .m8_startup_overhead import main as m8_main


def run_all():
    print("\n" + "=" * 70)
    print("  MTE (MEMORY TRANSFER ENGINE) MICROBENCHMARKS (M1-M8)")
    print("=" * 70)

    all_results = []

    print("\n>>> Running M1: L1 Buffer Read Bandwidth")
    results_m1 = m1_main()
    all_results.extend(results_m1)

    print("\n>>> Running M2: L1 Buffer Write Bandwidth")
    results_m2 = m2_main()
    all_results.extend(results_m2)

    print("\n>>> Running M3/M4/M5: L0 Buffer Bandwidths")
    results_m3 = m3_main()
    all_results.extend(results_m3)

    print("\n>>> Running M6: HBM Memory Latency")
    results_m6 = m6_main()
    all_results.extend(results_m6)

    print("\n>>> Running M7: Buffer Capacities")
    results_m7 = m7_main()
    all_results.extend(results_m7)

    print("\n>>> Running M8: Transfer Startup Overhead")
    results_m8 = m8_main()
    all_results.extend(results_m8)

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    save_results(all_results, os.path.join(output_dir, "mte_results.json"))

    print("\n" + "=" * 70)
    print("  MTE SUMMARY")
    print("=" * 70)
    for r in all_results:
        if "size=" not in (r.notes or ""):
            print(f"  [{r.param_id}] {r.param_name}: {r.value:.4f} {r.unit}")

    return all_results


if __name__ == "__main__":
    run_all()
