#!/usr/bin/env python3
"""
C5: Matrix Multiplication Latency Scaling with Size
Measures how matrix multiplication latency scales as matrix dimensions (M/N/K) grow.

Method: Sweep Matrix Dimensions
- Execute matrix multiplications with varying dimensions.
- Measure latency for each configuration.
- Plot latency vs. matrix size to reveal the scaling relationship.
- This reveals: tile-based execution granularity, memory hierarchy effects,
  and the transition from compute-bound to memory-bound regions.

Design Rationale:
- The Cube Unit processes matmul in fixed-size tiles.
- For matrices smaller than a tile, latency is dominated by fixed overhead.
- For matrices much larger than a tile, latency grows with O(M*N*K).
- The scaling curve reveals the effective tile size and pipeline behavior.
- This is presented as a line chart in the report.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from common.benchmark import AscendBenchmark, EventTimer, BenchResult, print_result
from common.utils import compute_matmul_flops, compute_flops

try:
    import torch
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


class MatMulScalingBench(AscendBenchmark):
    """C5: Measure matrix multiplication latency scaling with size."""

    def __init__(self, device_id=0):
        super().__init__(
            param_id="C5",
            param_name="不同规模矩阵乘延迟缩放关系 (MatMul Latency Scaling)",
            category="Cube Unit",
            device_id=device_id,
        )

    def bench_kernel(self, m=256, n=256, k=256) -> float:
        """Execute a matrix multiplication. Returns time in ms."""
        timer = EventTimer()

        if HAS_NPU:
            a = torch.randn(m, k, device="npu")
            b = torch.randn(k, n, device="npu")

            timer.record_start()
            c = torch.mm(a, b)
            timer.record_end()
            elapsed = timer.elapsed_ms()
        else:
            a = np.random.randn(m, k).astype(np.float32)
            b = np.random.randn(k, n).astype(np.float32)

            timer.record_start()
            c = a @ b
            timer.record_end()
            elapsed = timer.elapsed_ms()

        return elapsed

    def measure_scaling(self, sizes_m=None, sizes_n=None, sizes_k=None,
                        num_warmup=3, num_iters=10):
        """
        Measure latency across various dimension configurations.
        Returns results suitable for plotting a scaling curve.
        """
        if sizes_m is None:
            sizes_m = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

        results = []

        # Fix n=k=256, vary M
        print("\n--- Varying M (n=k=256) ---")
        for m in sizes_m:
            res = self.run(
                num_warmup=num_warmup,
                num_iterations=num_iters,
                bench_kwargs={"m": m, "n": 256, "k": 256},
            )
            flops = compute_matmul_flops(m, 256, 256)
            gflops = compute_flops(flops, res.value / 1000.0) / 1e9

            res.unit = "ms"
            res.value = res.value  # already in ms
            res.notes = f"m={m}, n=256, k=256, GFLOPS={gflops:.2f}"
            results.append(res)
            print_result(res)

        # Fix m=k=256, vary N
        print("\n--- Varying N (m=k=256) ---")
        for n in sizes_m:
            res = self.run(
                num_warmup=num_warmup,
                num_iterations=num_iters,
                bench_kwargs={"m": 256, "n": n, "k": 256},
            )
            flops = compute_matmul_flops(256, n, 256)
            gflops = compute_flops(flops, res.value / 1000.0) / 1e9

            res.unit = "ms"
            res.value = res.value
            res.notes = f"m=256, n={n}, k=256, GFLOPS={gflops:.2f}"
            results.append(res)
            print_result(res)

        # Fix m=n=256, vary K
        print("\n--- Varying K (m=n=256) ---")
        for k in sizes_m:
            res = self.run(
                num_warmup=num_warmup,
                num_iterations=num_iters,
                bench_kwargs={"m": 256, "n": 256, "k": k},
            )
            flops = compute_matmul_flops(256, 256, k)
            gflops = compute_flops(flops, res.value / 1000.0) / 1e9

            res.unit = "ms"
            res.value = res.value
            res.notes = f"m=256, n=256, k={k}, GFLOPS={gflops:.2f}"
            results.append(res)
            print_result(res)

        # Square matrices of increasing size
        print("\n--- Square Matrices (m=n=k) ---")
        for size in sizes_m:
            res = self.run(
                num_warmup=num_warmup,
                num_iterations=num_iters,
                bench_kwargs={"m": size, "n": size, "k": size},
            )
            flops = compute_matmul_flops(size, size, size)
            gflops = compute_flops(flops, res.value / 1000.0) / 1e9

            res.unit = "ms"
            res.value = res.value
            res.notes = f"m=n=k={size}, GFLOPS={gflops:.2f}"
            results.append(res)
            print_result(res)

        return results

    def export_scaling_data(self, results, filepath="c5_scaling.csv"):
        """Export scaling data for plotting in report."""
        with open(filepath, "w") as f:
            f.write("m,n,k,latency_ms,gflops\n")
            for r in results:
                parts = r.notes.split(",")
                m = int(parts[0].split("=")[1])
                n = int(parts[1].split("=")[1])
                k = int(parts[2].split("=")[1])
                gflops = float(parts[3].split("=")[1])
                f.write(f"{m},{n},{k},{r.value:.6f},{gflops:.2f}\n")
        print(f"[INFO] Scaling data exported to {filepath}")


def main():
    print("=" * 70)
    print("  C5: Matrix Multiplication Latency Scaling")
    print("  Method: Sweep Matrix Dimensions (M, N, K)")
    print("=" * 70)

    bench = MatMulScalingBench()
    results = bench.measure_scaling()

    # Export data for plotting
    bench.export_scaling_data(results, "c5_scaling.csv")

    print("\n[C5] Latency scaling measurement complete.")
    print("     Use the CSV data to plot latency vs. matrix size in the report.")

    return results


if __name__ == "__main__":
    main()
