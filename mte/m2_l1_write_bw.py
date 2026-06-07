#!/usr/bin/env python3
"""
M2: L1 Buffer Write Bandwidth
==============================
Measures the peak write bandwidth to L1 Buffer on Ascend NPU.

Design:
- Use zero_() or fill_() for write-only operations (no read path).
- Data sizes ≤512KB target L1-resident writes.
- After warmup, measure steady-state write throughput.
- Moderate batching to reduce overhead while preserving L1 BW accuracy.
"""

import sys, os
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

    def _get_batch(self, num_elements):
        size_kb = num_elements * 4 // 1024
        if size_kb <= 64: return 20
        if size_kb <= 256: return 10
        if size_kb <= 512: return 5
        if size_kb <= 4096: return 10
        if size_kb <= 16384: return 5
        return 2

    def bench_kernel(self, num_elements=1024 * 256) -> float:
        """
        Perform streaming writes to NPU memory (zero_).
        Returns per-operation time in milliseconds.
        """
        batch_size = self._get_batch(num_elements)

        if HAS_NPU:
            dst = torch.zeros(num_elements, device="npu")

            # Warmup
            for _ in range(30):
                dst.zero_()
            torch.npu.synchronize()

            start_e = torch.npu.Event(enable_timing=True)
            end_e = torch.npu.Event(enable_timing=True)

            start_e.record()
            for _ in range(batch_size):
                dst.zero_()
            end_e.record()
            torch.npu.synchronize()
            elapsed = start_e.elapsed_time(end_e) / batch_size
        else:
            dst = np.zeros(num_elements, dtype=np.float32)
            timer = EventTimer()
            timer.record_start()
            dst.fill(3.14)
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def measure_bandwidth(self, element_counts=None,
                          num_warmup=5, num_iters=50):
        if element_counts is None:
            element_counts = [
                4 * 1024, 8 * 1024, 16 * 1024, 32 * 1024,
                64 * 1024, 128 * 1024,
                256 * 1024, 1024 * 1024, 4 * 1024 * 1024,
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
            batch_size = self._get_batch(count)
            size_kb = total_bytes / 1024
            region = "L1" if size_kb <= 512 else "HBM"
            res.notes = f"elements={count}, size={size_kb:.0f}KB, write-only, batch={batch_size}, {region}"
            results.append(res)
            print_result(res)
        return results


def main():
    print("=" * 70)
    print("  M2: L1 Buffer Write Bandwidth")
    print("  Method: Streaming Write (zero_) with Warmup")
    print("=" * 70)

    bench = L1WriteBandwidthBench()
    results = bench.measure_bandwidth()

    l1_results = [r for r in results if 'L1' in r.notes]
    if l1_results:
        l1_bws = [r.value for r in l1_results]
        peak_l1 = max(l1_bws)
        print(f"\n[M2] Peak L1 Buffer write bandwidth: {peak_l1:.2f} GB/s")
    hbm_results = [r for r in results if 'HBM' in r.notes]
    if hbm_results:
        hbm_bw = max(r.value for r in hbm_results)
        print(f"     HBM write BW: {hbm_bw:.2f} GB/s")

    return results


if __name__ == "__main__":
    main()
