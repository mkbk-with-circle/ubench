#!/usr/bin/env python3
"""
M1: HBM/L1 Buffer Read Bandwidth — OPTIMIZED VERSION
=====================================================
Key optimizations over original:
1. Batched copies per measurement to amortize event/sync overhead
2. Pre-allocated events (reused across iterations)
3. More warmup iterations for thermal/frequency stabilization
4. Median-based statistics (outlier resistant)
5. Proper separation of HBM BW vs pipelined DMA BW
6. FP32 and FP16 tests
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from common.benchmark import BenchResult, print_result
from common.utils import compute_bandwidth_gbs

try:
    import torch
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


def measure_copy_bw(num_elements, batch_size=1, num_warmup=10, num_iters=50,
                    dtype=torch.float32, device='npu:0'):
    """
    Optimized bandwidth measurement with batched copies.
    Returns array of per-copy times in ms.
    """
    src = torch.randn(num_elements, dtype=dtype, device=device)
    dst = torch.zeros(num_elements, dtype=dtype, device=device)

    # Warmup
    for _ in range(num_warmup):
        dst.copy_(src)
    torch.npu.synchronize()

    # Pre-allocate events
    start_e = torch.npu.Event(enable_timing=True)
    end_e = torch.npu.Event(enable_timing=True)

    times = []
    for _ in range(num_iters):
        start_e.record()
        for _ in range(batch_size):
            dst.copy_(src)
        end_e.record()
        torch.npu.synchronize()
        times.append(start_e.elapsed_time(end_e) / batch_size)

    return np.array(times)


def run_m1_optimized(device='npu:0'):
    """Run optimized M1 benchmark."""
    print("=" * 70)
    print("  M1: HBM Read Bandwidth — OPTIMIZED")
    print("  Method: Batched copies with event timing")
    print("=" * 70)

    results = []

    # === Test 1: Size sweep with optimal batch ===
    print("\n--- Size sweep (FP32, optimal batch) ---")
    print(f"  {'Size':>8s} {'Batch':>6s} {'BW(GB/s)':>10s} {'Med(ms)':>10s} {'CV':>8s} {'Type':<15s}")

    size_configs = [
        (64,    200, 'L1-range'),
        (256,   200, 'L1-range'),
        (1024,  200, 'L1-range'),
        (4096,  100, 'L1-boundary'),
        (8192,   50, 'HBM-boundary'),
        (16384,  20, 'HBM'),
        (32768,  10, 'HBM'),
        (65536,   4, 'HBM'),
    ]

    for size_kb, batch, category in size_configs:
        n = size_kb * 1024 // 4  # FP32 elements
        total_bytes = size_kb * 1024

        times = measure_copy_bw(n, batch_size=batch, device=device)
        med = float(np.median(times))
        mean = float(np.mean(times))
        std = float(np.std(times))
        cv = std / mean if mean > 0 else 0
        bw = compute_bandwidth_gbs(total_bytes, med)

        # Element size for notes
        element_size = 4  # FP32

        res = BenchResult(
            param_id="M1",
            param_name="HBM读取带宽 (HBM Read Bandwidth)",
            category="MTE (Memory Transfer Engine)",
            value=bw,
            unit="GB/s",
            raw_times=times.tolist(),
            num_iterations=len(times),
            num_warmup=10,
            std_dev=std,
            cv=cv,
            median=med,
            p25=float(np.percentile(times, 25)),
            p75=float(np.percentile(times, 75)),
            notes=f"elements={n}, size={total_bytes/1024:.0f}KB, batch={batch}, {category}",
        )
        results.append(res)
        print(f"  {size_kb:>6d}KB {batch:>6d} {bw:10.2f} {med:10.4f} {cv:8.4f} {category}")

    # === Test 2: Pipelined DMA BW (medium size, large batch) ===
    print("\n--- Pipelined DMA bandwidth (8MB, large batches) ---")
    n_8mb = 8 * 1024 * 1024 // 4
    for batch in [10, 50, 100, 200]:
        times = measure_copy_bw(n_8mb, batch_size=batch, device=device)
        med = float(np.median(times))
        bw = compute_bandwidth_gbs(8 * 1024 * 1024, med)
        cv = float(np.std(times) / np.mean(times))
        print(f"  8MB batch={batch:>4d}: {bw:8.2f} GB/s, med={med:8.4f}ms, CV={cv:6.4f}")

    # === Test 3: FP16 bandwidth ===
    print("\n--- FP16 bandwidth ---")
    for size_kb in [4096, 16384]:
        n = size_kb * 1024 // 2  # FP16 elements
        total_bytes = size_kb * 1024
        batch = 100 if size_kb <= 8192 else 20

        times = measure_copy_bw(n, batch_size=batch, dtype=torch.float16, device=device)
        med = float(np.median(times))
        bw = compute_bandwidth_gbs(total_bytes, med)
        cv = float(np.std(times) / np.mean(times))
        print(f"  {size_kb}KB FP16 batch={batch}: {bw:8.2f} GB/s, med={med:8.4f}ms, CV={cv:6.4f}")

    # === Summary ===
    bws = [r.value for r in results]
    best_idx = np.argmax(bws)
    print(f"\n[M1 OPTIMIZED] Peak measured bandwidth: {results[best_idx].value:.2f} GB/s")
    print(f"     Config: {results[best_idx].notes}")
    print(f"\n  Ascend 310P3 HBM (LPDDR4X) theoretical: ~100-200 GB/s")
    print(f"  Measured efficiency: {results[best_idx].value/150*100:.1f}% of 150 GB/s")

    return results


if __name__ == "__main__":
    results = run_m1_optimized()
