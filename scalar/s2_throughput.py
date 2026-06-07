#!/usr/bin/env python3
"""
S2: Scalar Unit Throughput
Measures the maximum throughput of scalar arithmetic operations on Ascend NPU.

Method: Independent Operation Flood
- Issue a large number of INDEPENDENT scalar operations (no data dependencies).
- This allows the hardware to fully utilize instruction-level parallelism
  and pipeline overlapping, revealing the peak throughput.
- Throughput = total_operations / total_time

Design Rationale:
- Unlike S1 (latency), we remove all data dependencies so the scalar unit
  can issue operations at its maximum rate.
- By sweeping the number of operations, we find the saturation point
  where throughput stabilizes (pipeline fully occupied).
- The throughput reflects both the issue rate and the number of scalar pipelines.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from common.benchmark import AscendBenchmark, EventTimer, BenchResult, print_result
from common.utils import estimate_clock_freq_mhz

try:
    import torch
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


class ScalarThroughputBench(AscendBenchmark):
    """
    S2: Measure scalar unit throughput by issuing independent operations.

    Key insight: Independent operations can be executed in parallel
    by the scalar pipeline. We measure the steady-state throughput.
    """

    def __init__(self, device_id=0):
        super().__init__(
            param_id="S2",
            param_name="标量单元吞吐率 (Scalar Unit Throughput)",
            category="Scalar Unit",
            device_id=device_id,
        )

    def bench_kernel(self, num_ops=10000) -> float:
        """
        Execute many independent scalar operations.
        Returns elapsed time in milliseconds.
        Uses pre-allocated tensors to avoid allocation overhead.
        """
        if HAS_NPU:
            # Pre-allocate all tensors
            values = [torch.tensor([float(i)], device="npu") for i in range(num_ops)]
            results = [torch.zeros(1, device="npu") for _ in range(num_ops)]

            # Warmup
            for _ in range(5):
                for i in range(num_ops):
                    results[i].copy_(values[i] + values[i])
            torch.npu.synchronize()

            # Pre-allocate events
            start_e = torch.npu.Event(enable_timing=True)
            end_e = torch.npu.Event(enable_timing=True)

            start_e.record()
            for i in range(num_ops):
                results[i].copy_(values[i] + values[i])
            end_e.record()
            torch.npu.synchronize()
            elapsed = start_e.elapsed_time(end_e)
        else:
            values = [float(i) for i in range(num_ops)]
            timer = EventTimer()
            timer.record_start()
            results = []
            for i in range(num_ops):
                r = values[i] + values[i]
                results.append(r)
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def measure_throughput(self, op_counts=None, num_warmup=5, num_iters=30):
        """
        Measure throughput at different operation counts to find saturation.
        """
        if op_counts is None:
            op_counts = [100, 500, 1000, 5000, 10000, 50000, 100000]

        freq_mhz = estimate_clock_freq_mhz()
        results = []

        for count in op_counts:
            res = self.run(
                num_warmup=num_warmup,
                num_iterations=num_iters,
                bench_kwargs={"num_ops": count},
            )

            # ops per cycle
            cycles = res.value * freq_mhz * 1e3  # ms -> cycles
            ops_per_cycle = count / cycles if cycles > 0 else 0

            res.unit = "ops/cycle"
            res.value = ops_per_cycle
            res.notes = f"num_ops={count}, freq={freq_mhz}MHz"
            results.append(res)
            print_result(res)

        return results


def main():
    print("=" * 70)
    print("  S2: Scalar Unit Throughput")
    print("  Method: Independent Operation Flood")
    print("=" * 70)

    bench = ScalarThroughputBench()
    results = bench.measure_throughput()

    # Best throughput is max ops/cycle
    throughputs = [r.value for r in results]
    if throughputs:
        best = max(throughputs)
        print(f"\n[S2] Estimated scalar throughput: {best:.4f} ops/cycle")

    return results


if __name__ == "__main__":
    main()
