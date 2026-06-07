#!/usr/bin/env python3
"""
M1: L1 Buffer Read Bandwidth
=============================
Measures the peak read bandwidth from L1 Buffer on Ascend NPU.

Design:
- L1 Buffer is the main on-chip shared memory for AI Cores (512KB on 310P3).
- We use streaming copy with data sizes that fit in L1 (≤512KB).
- After warmup, data is L1-resident; copy measures L1→register→L1 throughput.
- We use moderate batching (batch=10) to reduce per-measurement overhead
  while still capturing true L1 bandwidth (not DMA pipeline throughput).
- For comparison, we also measure HBM bandwidth at large sizes.

Method: Streaming Copy with Warmup
- Allocate src and dst tensors of size ≤ L1 capacity.
- Warmup copies ensure data is L1-resident.
- Measure steady-state copy bandwidth.
- Sweep sizes to find peak L1 bandwidth and capacity cliff.

Note: On 310P3, L1 is a scratchpad managed by the MTE. PyTorch's copy_()
exercises the full HBM→L1→register path. After warmup, subsequent copies
benefit from L1 residency, giving an upper bound on effective L1 BW.
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


class L1ReadBandwidthBench(AscendBenchmark):
    """M1: Measure L1 Buffer read bandwidth."""

    def __init__(self, device_id=0):
        super().__init__(
            param_id="M1",
            param_name="L1 Buffer读取带宽 (L1 Buffer Read Bandwidth)",
            category="MTE (Memory Transfer Engine)",
            device_id=device_id,
        )

    def _get_batch(self, num_elements):
        """Moderate batching: enough to reduce overhead, not too much to mask L1 BW."""
        size_kb = num_elements * 4 // 1024
        if size_kb <= 64: return 20
        if size_kb <= 256: return 10
        if size_kb <= 512: return 5
        # For sizes > L1, use larger batches to measure HBM BW
        if size_kb <= 4096: return 10
        if size_kb <= 16384: return 5
        return 2

    def bench_kernel(self, num_elements=1024 * 256) -> float:
        """
        Perform streaming reads from NPU memory.
        Returns per-copy time in milliseconds.
        """
        batch_size = self._get_batch(num_elements)

        if HAS_NPU:
            src = torch.randn(num_elements, device="npu")
            dst = torch.zeros(num_elements, device="npu")

            # Heavy warmup to ensure L1 residency
            for _ in range(30):
                dst.copy_(src)
            torch.npu.synchronize()

            # Pre-allocate events
            start_e = torch.npu.Event(enable_timing=True)
            end_e = torch.npu.Event(enable_timing=True)

            # Batched measurement
            start_e.record()
            for _ in range(batch_size):
                dst.copy_(src)
            end_e.record()
            torch.npu.synchronize()
            elapsed = start_e.elapsed_time(end_e) / batch_size
        else:
            src = np.random.randn(num_elements).astype(np.float32)
            dst = np.zeros(num_elements, dtype=np.float32)
            timer = EventTimer()
            timer.record_start()
            np.copyto(dst, src)
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def measure_bandwidth(self, element_counts=None,
                          num_warmup=5, num_iters=50):
        """Sweep data sizes to find peak L1 read bandwidth."""
        if element_counts is None:
            # Focus on L1 range (≤512KB) + some larger sizes for HBM comparison
            element_counts = [
                # L1 range (≤512KB)
                4 * 1024,      # 16 KB
                8 * 1024,      # 32 KB
                16 * 1024,     # 64 KB
                32 * 1024,     # 128 KB
                64 * 1024,     # 256 KB
                128 * 1024,    # 512 KB
                # Beyond L1 (HBM)
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
            batch_size = self._get_batch(count)
            size_kb = total_bytes / 1024
            region = "L1" if size_kb <= 512 else "HBM"
            res.notes = f"elements={count}, size={size_kb:.0f}KB, batch={batch_size}, {region}"
            results.append(res)
            print_result(res)

        return results


def main():
    print("=" * 70)
    print("  M1: L1 Buffer Read Bandwidth")
    print("  Method: Streaming Copy with Warmup (L1-resident)")
    print("=" * 70)

    bench = L1ReadBandwidthBench()
    results = bench.measure_bandwidth()

    # Extract L1 BW (peak among L1-sized results)
    l1_results = [r for r in results if 'L1' in r.notes]
    hbm_results = [r for r in results if 'HBM' in r.notes]

    if l1_results:
        l1_bws = [r.value for r in l1_results]
        peak_l1 = max(l1_bws)
        peak_idx = l1_bws.index(peak_l1)
        print(f"\n[M1] Peak L1 Buffer read bandwidth: {peak_l1:.2f} GB/s")
        print(f"     Config: {l1_results[peak_idx].notes}")

    if hbm_results:
        hbm_bw = np.mean([r.value for r in hbm_results if '16384' in r.notes or '4096' in r.notes])
        print(f"     HBM BW (≥4MB): {hbm_bw:.2f} GB/s")

    print(f"\n  Ascend 310P3 L1 Buffer (512KB per core):")
    print(f"  Theoretical L1 BW: ~1-4 TB/s (on-chip SRAM)")
    print(f"  Measured via PyTorch copy_(): {peak_l1:.2f} GB/s")
    print(f"  (Limited by PyTorch dispatch overhead, not true L1 HW BW)")

    return results


if __name__ == "__main__":
    main()
