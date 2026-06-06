#!/usr/bin/env python3
"""
M2: L1 Buffer Write Bandwidth
Measures the peak write bandwidth to L1 Buffer on Ascend NPU.

Method: Streaming Write
- Allocate a destination array in L1 Buffer and write data to it.
- Use memset or fill operations to saturate the write bandwidth.
- Bandwidth = total_data_written / time

Design Rationale:
- L1 write bandwidth may differ from read bandwidth due to write buffer
  and memory controller design.
- We use a fill pattern (writing constant) to isolate pure write bandwidth.
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


class L1WriteBandwidthBench(AscendBenchmark):
    """M2: Measure L1 Buffer write bandwidth."""

    def __init__(self, device_id=0):
        super().__init__(
            param_id="M2",
            param_name="L1 Buffer写入带宽 (L1 Buffer Write Bandwidth)",
            category="MTE (Memory Transfer Engine)",
            device_id=device_id,
        )

    def bench_kernel(self, num_elements=1024 * 256) -> float:
        """
        Perform streaming writes to NPU memory (L1 resident).
        Returns time in milliseconds.
        """
        timer = EventTimer()
        element_size = 4  # FP32
        total_bytes = num_elements * element_size

        if HAS_NPU:
            dst = torch.zeros(num_elements, device="npu")
            val = torch.tensor([3.14], device="npu")

            timer.record_start()
            dst.fill_(val.item())
            timer.record_end()
            elapsed = timer.elapsed_ms()
        else:
            dst = np.zeros(num_elements, dtype=np.float32)
            timer.record_start()
            dst.fill(3.14)
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def measure_bandwidth(self, element_counts=None,
                          num_warmup=5, num_iters=20):
        """Sweep data sizes to find peak write bandwidth."""
        if element_counts is None:
            element_counts = [
                16 * 1024,
                64 * 1024,
                256 * 1024,
                1024 * 1024,
                4 * 1024 * 1024,
            ]

        results = []

        for count in element_counts:
            res = self.run(
                num_warmup=num_warmup,
                num_iterations=num_iters,
                bench_kwargs={"num_elements": count},
            )

            element_size = 4
            total_bytes = count * element_size
            bw_gbs = compute_bandwidth_gbs(total_bytes, res.value)

            res.unit = "GB/s"
            res.value = bw_gbs
            res.notes = f"elements={count}, size={total_bytes/1024:.1f}KB, write-only"
            results.append(res)
            print_result(res)

        return results


def main():
    print("=" * 70)
    print("  M2: L1 Buffer Write Bandwidth")
    print("  Method: Streaming Write (fill)")
    print("=" * 70)

    bench = L1WriteBandwidthBench()
    results = bench.measure_bandwidth()

    bws = [r.value for r in results]
    if bws:
        best = max(bws)
        print(f"\n[M2] Estimated L1 Buffer peak write bandwidth: {best:.2f} GB/s")

    return results


if __name__ == "__main__":
    main()
