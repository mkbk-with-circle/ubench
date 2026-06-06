#!/usr/bin/env python3
"""
Quick validation script to test that all benchmarks can at least be imported.
Does not require actual NPU hardware.
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ubench"))

def test_import(module_name: str) -> bool:
    """Test that a module can be imported successfully."""
    try:
        __import__(module_name)
        print(f"  ✅ {module_name}")
        return True
    except Exception as e:
        print(f"  ❌ {module_name}: {e}")
        return False


def main():
    print("=" * 50)
    print("  Ascend NPU μbench Import Test")
    print("=" * 50)

    modules = [
        # Common
        "common.benchmark",
        "common.utils",
        # Scalar
        "scalar.s1_latency",
        "scalar.s2_throughput",
        "scalar.s3_mem_latency",
        # Vector
        "vector.v1_add_latency",
        "vector.v2_mul_latency",
        "vector.v3_throughput",
        "vector.v4_pipeline_depth",
        "vector.v5_reg_latency",
        # Cube
        "cube.c1_tile_latency",
        "cube.c2_throughput",
        "cube.c3_pipeline_depth",
        "cube.c4_l0_latency",
        "cube.c5_scaling",
        # MTE
        "mte.m1_l1_read_bw",
        "mte.m2_l1_write_bw",
        "mte.m3_m4_m5_l0_bw",
        "mte.m6_hbm_latency",
        "mte.m7_buffer_capacity",
        "mte.m8_startup_overhead",
    ]

    passed = 0
    failed = 0

    for mod in modules:
        if test_import(mod):
            passed += 1
        else:
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
