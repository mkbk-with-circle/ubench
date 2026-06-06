#!/usr/bin/env python3
"""
V5: Vector Register Access Latency
Measures the read/write latency of vector registers on the Ascend NPU.

Method: Register File Pressure Test
- The vector unit has a register file (typically 32 or 64 vector registers).
- We measure the access latency by creating register pressure:
  a large number of live values forces register spilling/reloading.
- By varying the number of live registers between operations, we can
  distinguish register file access latency from compute latency.

Design Rationale:
- Vector registers on DaVinci hold short vectors (e.g., 32 FP32 elements).
- Register file access adds latency to each instruction.
- By comparing operations with minimal register usage vs. maximum register
  usage, we can estimate the extra latency from register file access.
- Read vs. write latency may differ (write typically has bypass).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from common.benchmark import AscendBenchmark, EventTimer, BenchResult, print_result
from common.utils import estimate_clock_freq_mhz, ms_to_cycles

try:
    import torch
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


class VectorRegAccessLatencyBench(AscendBenchmark):
    """V5: Measure vector register read/write latency."""

    def __init__(self, device_id=0):
        super().__init__(
            param_id="V5",
            param_name="向量寄存器访问延迟 (Vector Register Access Latency)",
            category="Vector Unit",
            device_id=device_id,
        )

    def bench_kernel_read(self, num_regs=1, vec_size=256, chain_length=100) -> float:
        """
        Measure read latency: use many source registers.
        More source registers = more read ports exercised = higher read pressure.
        """
        timer = EventTimer()

        if HAS_NPU:
            # Create N source registers worth of data
            sources = [torch.randn(vec_size, device="npu") for _ in range(num_regs)]
            result = torch.zeros(vec_size, device="npu")

            timer.record_start()
            for _ in range(chain_length):
                # Accumulate from many source registers
                acc = sources[0]
                for j in range(1, num_regs):
                    acc = acc + sources[j]
                result = result + acc
            timer.record_end()
            elapsed = timer.elapsed_ms()
        else:
            sources = [np.random.randn(vec_size).astype(np.float32) for _ in range(num_regs)]
            result = np.zeros(vec_size, dtype=np.float32)

            timer.record_start()
            for _ in range(chain_length):
                acc = sources[0]
                for j in range(1, num_regs):
                    acc = acc + sources[j]
                result = result + acc
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def bench_kernel_write(self, num_regs=1, vec_size=256, chain_length=100) -> float:
        """
        Measure write latency: write to many destination registers.
        More destinations = more write port pressure.
        """
        timer = EventTimer()

        if HAS_NPU:
            a = torch.ones(vec_size, device="npu")
            dests = [torch.zeros(vec_size, device="npu") for _ in range(num_regs)]

            timer.record_start()
            for _ in range(chain_length):
                # Write to many destinations
                for j in range(num_regs):
                    dests[j] = dests[j] + a
            timer.record_end()
            elapsed = timer.elapsed_ms()
        else:
            a = np.ones(vec_size, dtype=np.float32)
            dests = [np.zeros(vec_size, dtype=np.float32) for _ in range(num_regs)]

            timer.record_start()
            for _ in range(chain_length):
                for j in range(num_regs):
                    dests[j] = dests[j] + a
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def measure_latency(self, reg_counts=None, num_warmup=3, num_iters=20):
        """Measure read and write latency at various register pressures."""
        if reg_counts is None:
            reg_counts = [1, 2, 4, 8, 16, 32]

        freq_mhz = estimate_clock_freq_mhz()
        results = []

        # Measure read latency
        print("\n--- Vector Register READ Latency ---")
        for nr in reg_counts:
            res = self.run(
                num_warmup=num_warmup,
                num_iterations=num_iters,
                bench_kwargs={"num_regs": nr},
                bench_args=(nr, 256, 100),
            )

            # Override: use bench_kernel_read manually
            raw_times = []
            for _ in range(num_iters):
                t = self.bench_kernel_read(num_regs=nr)
                raw_times.append(t)

            arr = np.array(raw_times)
            mean_t = float(np.mean(arr))
            cycles = ms_to_cycles(mean_t, freq_mhz)
            cycles_per_op_read = cycles / (100 * nr)  # chain_length * num_regs

            from common.benchmark import BenchResult
            res = BenchResult(
                param_id="V5_READ",
                param_name=f"向量寄存器读延迟 (num_regs={nr})",
                category="Vector Unit",
                value=cycles_per_op_read,
                unit="cycles",
                raw_times=raw_times,
                num_iterations=num_iters,
                num_warmup=num_warmup,
                std_dev=float(np.std(arr)),
                cv=float(np.std(arr)) / mean_t if mean_t > 0 else 0,
                median=float(np.median(arr)),
                p25=float(np.percentile(arr, 25)),
                p75=float(np.percentile(arr, 75)),
                notes=f"READ, num_regs={nr}, freq={freq_mhz}MHz",
            )
            results.append(res)
            print_result(res)

        # Measure write latency
        print("\n--- Vector Register WRITE Latency ---")
        for nr in reg_counts:
            raw_times = []
            for _ in range(num_iters):
                t = self.bench_kernel_write(num_regs=nr)
                raw_times.append(t)

            arr = np.array(raw_times)
            mean_t = float(np.mean(arr))
            cycles = ms_to_cycles(mean_t, freq_mhz)
            cycles_per_op_write = cycles / (100 * nr)

            from common.benchmark import BenchResult
            res = BenchResult(
                param_id="V5_WRITE",
                param_name=f"向量寄存器写延迟 (num_regs={nr})",
                category="Vector Unit",
                value=cycles_per_op_write,
                unit="cycles",
                raw_times=raw_times,
                num_iterations=num_iters,
                num_warmup=num_warmup,
                std_dev=float(np.std(arr)),
                cv=float(np.std(arr)) / mean_t if mean_t > 0 else 0,
                median=float(np.median(arr)),
                p25=float(np.percentile(arr, 25)),
                p75=float(np.percentile(arr, 75)),
                notes=f"WRITE, num_regs={nr}, freq={freq_mhz}MHz",
            )
            results.append(res)
            print_result(res)

        return results


def main():
    print("=" * 70)
    print("  V5: Vector Register Access Latency")
    print("  Method: Register File Pressure Test (Read/Write)")
    print("=" * 70)

    bench = VectorRegAccessLatencyBench()
    results = bench.measure_latency()

    # Summarize
    reads = [r for r in results if "READ" in r.notes]
    writes = [r for r in results if "WRITE" in r.notes]

    if reads:
        best_read = min(r.value for r in reads)
        print(f"\n[V5] Estimated vector register READ latency: {best_read:.2f} cycles")
    if writes:
        best_write = min(r.value for r in writes)
        print(f"[V5] Estimated vector register WRITE latency: {best_write:.2f} cycles")

    return results


if __name__ == "__main__":
    main()
