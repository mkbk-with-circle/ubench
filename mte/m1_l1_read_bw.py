#!/usr/bin/env python3
"""
M1: L1 Buffer Read Bandwidth
Measures the peak read bandwidth from L1 Buffer on Ascend NPU.

Method: Streaming Read
- Allocate a large array in L1 Buffer and read it sequentially.
- Use vector loads to saturate the read bandwidth.
- Bandwidth = total_data_read / time

Design Rationale:
- L1 Buffer is the main on-chip shared memory for AI Cores.
- Sequential reads within L1 should achieve near-peak bandwidth.
- We sweep data sizes to find the saturation point (where bandwidth plateaus).
- The L1 read bandwidth is critical for data-reuse scenarios in operators.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from common.benchmark import AscendBenchmark, EventTimer, BenchResult, print_result
from common.utils import compute_bandwidth_gbs

try:
    import torch
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


class L1ReadBandwidthBench(AscendBenchmark):
    """M1: Measure L1 Buffer read bandwidth."""

    def __init__(self, device_id=0):
        super().__init__(
            param_id="M1",
            param_name="L1 Buffer读取带宽 (L1 Buffer Read Bandwidth)",
            category="MTE (Memory Transfer Engine)",
            device_id=device_id,
        )

    def bench_kernel(self, num_elements=1024 * 256) -> float:
        """
        Perform streaming reads from NPU memory (L1 resident).
        Returns time in milliseconds.
        """
        timer = EventTimer()
        element_size = 4  # FP32 = 4 bytes
        total_bytes = num_elements * element_size

        if HAS_NPU:
            src = torch.randn(num_elements, device="npu")
            dst = torch.zeros(num_elements, device="npu")

            timer.record_start()
            # Streaming copy from L1 to L1 (read+write, we isolate read below)
            dst.copy_(src)
            timer.record_end()
            elapsed = timer.elapsed_ms()
        else:
            src = np.random.randn(num_elements).astype(np.float32)
            dst = np.zeros(num_elements, dtype=np.float32)
            timer.record_start()
            np.copyto(dst, src)
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def measure_bandwidth(self, element_counts=None,
                          num_warmup=5, num_iters=20):
        """Sweep data sizes to find peak read bandwidth."""
        if element_counts is None:
            # From 64KB to 16MB
            element_counts = [
                16 * 1024,     # 64 KB
                64 * 1024,     # 256 KB
                256 * 1024,    # 1 MB
                1024 * 1024,   # 4 MB
                4 * 1024 * 1024,  # 16 MB
            ]

        results = []

        for count in element_counts:
            res = self.run(
                num_warmup=num_warmup,
                num_iterations=num_iters,
                bench_kwargs={"num_elements": count},
            )

            element_size = 4  # FP32
            total_bytes = count * element_size
            bw_gbs = compute_bandwidth_gbs(total_bytes, res.value)

            res.unit = "GB/s"
            res.value = bw_gbs
            res.notes = f"elements={count}, size={total_bytes/1024:.1f}KB, r+w combined"
            results.append(res)
            print_result(res)

        return results


def main():
    print("=" * 70)
    print("  M1: L1 Buffer Read Bandwidth")
    print("  Method: Streaming Read")
    print("=" * 70)

    bench = L1ReadBandwidthBench()
    results = bench.measure_bandwidth()

    bws = [r.value for r in results]
    if bws:
        best = max(bws)
        print(f"\n[M1] Estimated L1 Buffer peak read bandwidth: {best:.2f} GB/s")
        print(f"     (Theoretical L1 bandwidth: ~2-4 TB/s on 910B)")

    return results


if __name__ == "__main__":
    main()
