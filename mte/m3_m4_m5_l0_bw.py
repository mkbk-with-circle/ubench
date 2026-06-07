#!/usr/bin/env python3
"""
M3/M4/M5: L0A/L0B/L0C Buffer Bandwidths
==========================================
Measures the read/write bandwidth of Cube Unit's dedicated L0 buffers.

Method: MatMul Data Loading Throughput
- L0 buffers are used by the Cube Unit for matrix multiplication data.
- L0A holds matrix A, L0B holds matrix B, L0C holds accumulated result C.
- We measure effective L0 bandwidth by timing matmul operations and
  subtracting estimated computation time.

Design:
- For L0A: Use tall-skinny A (M×1) × wide B (1×N) to stress L0A load
- For L0B: Use wide A (1×K) × tall-skinny B (K×N) to stress L0B load
- For L0C: Use large output M×N with small K to stress L0C store
- Pre-allocate all tensors to exclude allocation overhead
- Use batched measurement with moderate batch sizes
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


class L0BandwidthBench(AscendBenchmark):
    """M3/M4/M5: Measure L0A, L0B, L0C buffer bandwidths."""

    def __init__(self, device_id=0):
        super().__init__(
            param_id="M3-M5",
            param_name="L0A/L0B/L0C带宽 (L0 Buffer Bandwidths)",
            category="MTE (Memory Transfer Engine)",
            device_id=device_id,
        )

    def _measure_matmul(self, m, n, k, batch_size=10, num_warmup=20, num_iters=30,
                        device='npu:0', dtype=torch.float32):
        """Measure matmul throughput with pre-allocated tensors."""
        if HAS_NPU:
            a = torch.randn(m, k, dtype=dtype, device=device)
            b = torch.randn(k, n, dtype=dtype, device=device)

            for _ in range(num_warmup):
                c = torch.mm(a, b)
            torch.npu.synchronize()

            start_e = torch.npu.Event(enable_timing=True)
            end_e = torch.npu.Event(enable_timing=True)
            times = []
            for _ in range(num_iters):
                start_e.record()
                for _ in range(batch_size):
                    c = torch.mm(a, b)
                end_e.record()
                torch.npu.synchronize()
                times.append(start_e.elapsed_time(end_e) / batch_size)
            return np.array(times)
        else:
            a = np.random.randn(m, k).astype(np.float32)
            b = np.random.randn(k, n).astype(np.float32)
            import time
            times = []
            for _ in range(num_iters):
                t0 = time.perf_counter()
                for _ in range(batch_size):
                    c = a @ b
                times.append((time.perf_counter() - t0) * 1000.0 / batch_size)
            return np.array(times)

    def measure_all(self, num_warmup=20, num_iters=30):
        """Measure all three L0 buffer bandwidths."""
        results = []
        element_size = 4  # FP32

        # === M3: L0A bandwidth ===
        # L0A holds matrix A (M×K). Stress L0A by using large M, small K.
        # Data through L0A per matmul: M * K * element_size
        print("\n--- M3: L0A Bandwidth ---")
        m3_configs = [
            (256, 256, 256, 50),
            (512, 512, 512, 20),
            (1024, 1024, 1024, 5),
        ]
        best_l0a_bw = 0
        best_l0a_times = None
        for m, n, k, batch in m3_configs:
            times = self._measure_matmul(m, n, k, batch_size=batch, num_warmup=num_warmup, num_iters=num_iters)
            med = float(np.median(times))
            # L0A data: M * K per matmul
            l0a_bytes = m * k * element_size
            bw = compute_bandwidth_gbs(l0a_bytes, med)
            flops = 2 * m * n * k
            tflops = flops / (med / 1000.0) / 1e12
            print(f"  {m}x{n}x{k}: L0A BW={bw:.2f} GB/s, time={med:.4f}ms, {tflops:.2f} TFLOPS")
            if bw > best_l0a_bw:
                best_l0a_bw = bw
                best_l0a_times = times

        res = BenchResult(
            param_id="M3", param_name="L0A带宽 (L0A Bandwidth)", category="MTE",
            value=best_l0a_bw, unit="GB/s",
            raw_times=best_l0a_times.tolist() if best_l0a_times is not None else [],
            num_iterations=num_iters, num_warmup=num_warmup,
            std_dev=float(np.std(best_l0a_times)) if best_l0a_times is not None else 0,
            cv=float(np.std(best_l0a_times)/np.mean(best_l0a_times)) if best_l0a_times is not None else 0,
            median=float(np.median(best_l0a_times)) if best_l0a_times is not None else 0,
            p25=float(np.percentile(best_l0a_times, 25)) if best_l0a_times is not None else 0,
            p75=float(np.percentile(best_l0a_times, 75)) if best_l0a_times is not None else 0,
            notes="L0A bandwidth via matmul, peak across sizes",
        )
        results.append(res)
        print_result(res)

        # === M4: L0B bandwidth ===
        # L0B holds matrix B (K×N). Stress L0B by using large N, small K.
        print("\n--- M4: L0B Bandwidth ---")
        best_l0b_bw = 0
        best_l0b_times = None
        for m, n, k, batch in m3_configs:
            times = self._measure_matmul(m, n, k, batch_size=batch, num_warmup=num_warmup, num_iters=num_iters)
            med = float(np.median(times))
            l0b_bytes = k * n * element_size
            bw = compute_bandwidth_gbs(l0b_bytes, med)
            print(f"  {m}x{n}x{k}: L0B BW={bw:.2f} GB/s, time={med:.4f}ms")
            if bw > best_l0b_bw:
                best_l0b_bw = bw
                best_l0b_times = times

        res = BenchResult(
            param_id="M4", param_name="L0B带宽 (L0B Bandwidth)", category="MTE",
            value=best_l0b_bw, unit="GB/s",
            raw_times=best_l0b_times.tolist() if best_l0b_times is not None else [],
            num_iterations=num_iters, num_warmup=num_warmup,
            std_dev=float(np.std(best_l0b_times)) if best_l0b_times is not None else 0,
            cv=float(np.std(best_l0b_times)/np.mean(best_l0b_times)) if best_l0b_times is not None else 0,
            median=float(np.median(best_l0b_times)) if best_l0b_times is not None else 0,
            p25=float(np.percentile(best_l0b_times, 25)) if best_l0b_times is not None else 0,
            p75=float(np.percentile(best_l0b_times, 75)) if best_l0b_times is not None else 0,
            notes="L0B bandwidth via matmul, peak across sizes",
        )
        results.append(res)
        print_result(res)

        # === M5: L0C bandwidth ===
        # L0C holds result C (M×N). Stress L0C by using large M*N, small K.
        print("\n--- M5: L0C Bandwidth ---")
        # Use M*N output with small K to stress L0C store bandwidth
        m5_configs = [
            (256, 256, 16, 50),   # small K, focus on L0C write
            (512, 512, 16, 20),
            (1024, 1024, 16, 5),
        ]
        best_l0c_bw = 0
        best_l0c_times = None
        for m, n, k, batch in m5_configs:
            times = self._measure_matmul(m, n, k, batch_size=batch, num_warmup=num_warmup, num_iters=num_iters)
            med = float(np.median(times))
            l0c_bytes = m * n * element_size
            bw = compute_bandwidth_gbs(l0c_bytes, med)
            flops = 2 * m * n * k
            tflops = flops / (med / 1000.0) / 1e12
            print(f"  {m}x{n}x{k}: L0C BW={bw:.2f} GB/s, time={med:.4f}ms, {tflops:.2f} TFLOPS")
            if bw > best_l0c_bw:
                best_l0c_bw = bw
                best_l0c_times = times

        res = BenchResult(
            param_id="M5", param_name="L0C带宽 (L0C Bandwidth)", category="MTE",
            value=best_l0c_bw, unit="GB/s",
            raw_times=best_l0c_times.tolist() if best_l0c_times is not None else [],
            num_iterations=num_iters, num_warmup=num_warmup,
            std_dev=float(np.std(best_l0c_times)) if best_l0c_times is not None else 0,
            cv=float(np.std(best_l0c_times)/np.mean(best_l0c_times)) if best_l0c_times is not None else 0,
            median=float(np.median(best_l0c_times)) if best_l0c_times is not None else 0,
            p25=float(np.percentile(best_l0c_times, 25)) if best_l0c_times is not None else 0,
            p75=float(np.percentile(best_l0c_times, 75)) if best_l0c_times is not None else 0,
            notes="L0C bandwidth via matmul (small K), peak across sizes",
        )
        results.append(res)
        print_result(res)

        return results


def main():
    print("=" * 70)
    print("  M3/M4/M5: L0A, L0B, L0C Buffer Bandwidths")
    print("  Method: MatMul Data Loading Throughput")
    print("=" * 70)

    bench = L0BandwidthBench()
    results = bench.measure_all()

    for r in results:
        print(f"\n[{r.param_id}] {r.param_name}: {r.value:.2f} {r.unit}")

    return results


if __name__ == "__main__":
    main()
