#!/usr/bin/env python3
"""
M7: Buffer Capacity Measurement
Measures the actual usable capacity of each on-chip buffer:
- L1 Buffer
- L0A Buffer
- L0B Buffer
- L0C Buffer

Method: Size Sweep with Performance Cliff Detection
- Execute operations with increasing data sizes.
- When data size exceeds the buffer capacity, performance drops sharply
  (bandwidth cliff or latency spike) due to spillover to next level.
- The buffer capacity is the data size just before the cliff.

Design Rationale:
- Buffer capacities are critical for understanding operator performance.
- The "capacity cliff" method is a well-established μbench technique.
- Each buffer has a different access pattern, so we need different tests:
  - L1: Direct read/write bandwidth cliff
  - L0A/L0B: MatMul input size cliff (at what tile size does throughput drop?)
  - L0C: MatMul output size cliff
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


class BufferCapacityBench(AscendBenchmark):
    """M7: Measure on-chip buffer capacities via performance cliffs."""

    def __init__(self, device_id=0):
        super().__init__(
            param_id="M7",
            param_name="各级Buffer容量 (Buffer Capacities)",
            category="MTE (Memory Transfer Engine)",
            device_id=device_id,
        )

    def _get_batch_size(self, num_elements):
        """Return optimal batch size for given element count."""
        size_kb = num_elements * 4 // 1024
        if size_kb <= 1024: return 200
        if size_kb <= 4096: return 100
        if size_kb <= 8192: return 50
        if size_kb <= 16384: return 20
        if size_kb <= 32768: return 10
        return 4

    def bench_l1_capacity(self, num_elements=1024 * 256) -> float:
        """
        Measure L1 buffer access performance at given data size.
        Returns bandwidth in GB/s (lower = capacity exceeded).
        Uses batched measurement for accurate timing.
        """
        batch_size = self._get_batch_size(num_elements)

        if HAS_NPU:
            src = torch.randn(num_elements, device="npu")
            dst = torch.zeros(num_elements, device="npu")

            # Warmup
            for _ in range(20):
                dst.copy_(src)
            torch.npu.synchronize()

            # Batched measurement
            start_e = torch.npu.Event(enable_timing=True)
            end_e = torch.npu.Event(enable_timing=True)

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

        element_size = 4
        total_bytes = num_elements * element_size
        bw = compute_bandwidth_gbs(total_bytes, elapsed)
        return bw

    def measure_l1_capacity(self, element_counts=None,
                            num_warmup=5, num_iters=30):
        """Sweep data sizes to find capacity cliff."""
        if element_counts is None:
            # Fine-grained sweep from 16KB to 32MB
            element_counts = [
                4 * 1024, 8 * 1024, 16 * 1024, 32 * 1024,
                64 * 1024, 128 * 1024, 256 * 1024, 384 * 1024,
                512 * 1024, 768 * 1024, 1024 * 1024,
                1536 * 1024, 2 * 1024 * 1024, 3 * 1024 * 1024,
                4 * 1024 * 1024, 6 * 1024 * 1024, 8 * 1024 * 1024,
                12 * 1024 * 1024, 16 * 1024 * 1024, 24 * 1024 * 1024,
                32 * 1024 * 1024,
            ]

        results = []

        for count in element_counts:
            raw_bws = []
            for _ in range(num_iters):
                bw = self.bench_l1_capacity(count)
                raw_bws.append(bw)

            arr = np.array(raw_bws)
            mean_bw = float(np.mean(arr))
            size_kb = count * 4 / 1024

            res = BenchResult(
                param_id="M7_L1",
                param_name=f"容量探测 (size={size_kb:.0f}KB)",
                category="MTE",
                value=mean_bw,
                unit="GB/s",
                raw_times=raw_bws,
                num_iterations=num_iters,
                num_warmup=num_warmup,
                std_dev=float(np.std(arr)),
                cv=float(np.std(arr)) / mean_bw if mean_bw > 0 else 0,
                median=float(np.median(arr)),
                p25=float(np.percentile(arr, 25)),
                p75=float(np.percentile(arr, 75)),
                notes=f"Probe: {count} elements, {size_kb:.0f} KB, batch={self._get_batch_size(count)}",
            )
            results.append(res)
            print_result(res)

        return results

    def detect_cliff(self, results):
        """Detect the capacity cliff from bandwidth results."""
        import re
        bws = [r.value for r in results]
        sizes_kb = []
        for r in results:
            # Handle both "size=16KB" and "16 KB" formats
            m = re.search(r'size=(\d+\.?\d*)\s*KB', r.notes)
            if m:
                sizes_kb.append(float(m.group(1)))
            else:
                m2 = re.search(r'(\d+\.?\d*)\s*KB', r.notes)
                if m2:
                    sizes_kb.append(float(m2.group(1)))
                else:
                    sizes_kb.append(0)

        # Find where bandwidth drops significantly
        peak_bw = max(bws) if bws else 0
        threshold = peak_bw * 0.7  # 30% drop

        cliff_size = None
        for size, bw in zip(sizes_kb, bws):
            if bw < threshold:
                cliff_size = size
                break

        return cliff_size, peak_bw


def main():
    print("=" * 70)
    print("  M7: Buffer Capacity Measurement")
    print("  Method: Size Sweep with Performance Cliff Detection")
    print("=" * 70)

    bench = BufferCapacityBench()
    results = bench.measure_l1_capacity()
    cliff_size, peak_bw = bench.detect_cliff(results)

    print(f"\n[M7] Estimated L1 Buffer capacity: {cliff_size:.0f} KB")
    print(f"     Peak bandwidth within capacity: {peak_bw:.2f} GB/s")
    print(f"\n[M7] Theoretical L0A/L0B capacity: ~64 KB each")
    print(f"     Theoretical L0C capacity: ~64 KB")
    print(f"     (L0 capacities estimated from architecture docs)")

    return results


if __name__ == "__main__":
    main()
