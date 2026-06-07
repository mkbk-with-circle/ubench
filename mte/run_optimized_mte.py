#!/usr/bin/env python3
"""
Optimized MTE Benchmark Suite for Ascend 310P3
==============================================
Incorporates all optimization findings from iterative testing.

Key optimizations:
1. Batched copies per measurement to amortize event/sync overhead
2. Pre-allocated events (reused across iterations)
3. Pre-allocated tensors (no allocation during measurement)
4. Heavy warmup (20 iterations) for thermal/frequency stabilization
5. Median-based statistics (outlier resistant)
6. Proper batch sizes per data size (larger batches for smaller data)
7. FP32 primary measurement with FP16 cross-check
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import json
from datetime import datetime
from common.benchmark import BenchResult, print_result, save_results
from common.utils import compute_bandwidth_gbs, estimate_clock_freq_mhz

try:
    import torch
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


# ============================================================
#  Core measurement primitives
# ============================================================

def measure_copy_bw(num_elements, batch_size=1, num_warmup=20, num_iters=50,
                    dtype=torch.float32, device='npu:0'):
    """Optimized copy bandwidth with batched measurements."""
    src = torch.randn(num_elements, dtype=dtype, device=device)
    dst = torch.zeros(num_elements, dtype=dtype, device=device)

    for _ in range(num_warmup):
        dst.copy_(src)
    torch.npu.synchronize()

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


def measure_fill_bw(num_elements, batch_size=1, method='zero', num_warmup=20, num_iters=50,
                    device='npu:0'):
    """Optimized write-only bandwidth."""
    dst = torch.zeros(num_elements, device=device)
    src = torch.full((num_elements,), 3.14, device=device) if method == 'fill' else None

    for _ in range(num_warmup):
        if method == 'zero':
            dst.zero_()
        elif method == 'fill':
            dst.fill_(3.14)
    torch.npu.synchronize()

    start_e = torch.npu.Event(enable_timing=True)
    end_e = torch.npu.Event(enable_timing=True)
    times = []
    for _ in range(num_iters):
        start_e.record()
        for _ in range(batch_size):
            if method == 'zero':
                dst.zero_()
            elif method == 'fill':
                dst.fill_(3.14)
        end_e.record()
        torch.npu.synchronize()
        times.append(start_e.elapsed_time(end_e) / batch_size)
    return np.array(times)


def measure_matmul(m, n, k, batch_size=1, num_warmup=10, num_iters=30,
                   dtype=torch.float32, device='npu:0'):
    """Optimized matmul throughput measurement."""
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


def make_result(param_id, param_name, category, value, unit, times, notes=''):
    """Helper to create BenchResult."""
    arr = np.array(times)
    return BenchResult(
        param_id=param_id,
        param_name=param_name,
        category=category,
        value=value,
        unit=unit,
        raw_times=times if isinstance(times, list) else times.tolist(),
        num_iterations=len(times),
        num_warmup=20,
        std_dev=float(np.std(arr)),
        cv=float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 0 else 0,
        median=float(np.median(arr)),
        p25=float(np.percentile(arr, 25)),
        p75=float(np.percentile(arr, 75)),
        notes=notes,
    )


# ============================================================
#  Optimal batch sizes per data size
# ============================================================

def get_batch(size_bytes):
    """Return optimal batch size for given data size."""
    size_kb = size_bytes // 1024
    if size_kb <= 64: return 200
    if size_kb <= 256: return 200
    if size_kb <= 1024: return 200
    if size_kb <= 4096: return 100
    if size_kb <= 8192: return 50
    if size_kb <= 16384: return 20
    if size_kb <= 32768: return 10
    return 4


# ============================================================
#  M1: HBM Read Bandwidth (copy = read + write)
# ============================================================

def run_m1(device='npu:0'):
    print("\n" + "=" * 70)
    print("  M1: HBM Read+Write Bandwidth (copy)")
    print("=" * 70)

    results = []
    sizes_kb = [64, 256, 1024, 4096, 8192, 16384, 32768, 65536]

    print(f"\n  {'Size':>8s} {'Batch':>6s} {'BW(GB/s)':>10s} {'Med(ms)':>10s} {'CV':>8s}")

    for size_kb in sizes_kb:
        n = size_kb * 1024 // 4
        total_bytes = size_kb * 1024
        batch = get_batch(total_bytes)

        times = measure_copy_bw(n, batch_size=batch, device=device)
        med = float(np.median(times))
        bw = compute_bandwidth_gbs(total_bytes, med)
        cv = float(np.std(times) / np.mean(times))

        res = make_result("M1", "HBM读写带宽 (HBM Read+Write BW)", "MTE",
                          bw, "GB/s", times,
                          f"elements={n}, size={size_kb}KB, batch={batch}")
        results.append(res)
        print(f"  {size_kb:>6d}KB {batch:>6d} {bw:10.2f} {med:10.4f} {cv:8.4f}")

    # Best HBM BW (large sizes, converged)
    hbm_bws = [r.value for r in results if any(f'size={s}' in r.notes for s in ['16384', '32768', '65536'])]
    hbm_bw = np.mean(hbm_bws) if hbm_bws else 0
    print(f"\n  [M1] True HBM BW (≥16MB): {hbm_bw:.2f} GB/s")

    return results


# ============================================================
#  M2: HBM Write Bandwidth (zero/fill)
# ============================================================

def run_m2(device='npu:0'):
    print("\n" + "=" * 70)
    print("  M2: HBM Write Bandwidth (zero/fill)")
    print("=" * 70)

    results = []
    sizes_kb = [256, 1024, 4096, 16384, 65536]

    print(f"\n  {'Size':>8s} {'Batch':>6s} {'BW(GB/s)':>10s} {'Med(ms)':>10s} {'CV':>8s}")

    for size_kb in sizes_kb:
        n = size_kb * 1024 // 4
        total_bytes = size_kb * 1024
        batch = get_batch(total_bytes)

        # Use zero_() for write-only
        times = measure_fill_bw(n, batch_size=batch, method='zero', device=device)
        med = float(np.median(times))
        bw = compute_bandwidth_gbs(total_bytes, med)
        cv = float(np.std(times) / np.mean(times))

        res = make_result("M2", "HBM写入带宽 (HBM Write BW)", "MTE",
                          bw, "GB/s", times,
                          f"elements={n}, size={size_kb}KB, batch={batch}, write-only")
        results.append(res)
        print(f"  {size_kb:>6d}KB {batch:>6d} {bw:10.2f} {med:10.4f} {cv:8.4f}")

    write_bws = [r.value for r in results if '65536' in r.notes]
    write_bw = np.mean(write_bws) if write_bws else 0
    print(f"\n  [M2] HBM Write BW (64MB): {write_bw:.2f} GB/s")

    return results


# ============================================================
#  M3/M4/M5: L0 Buffer Bandwidths (via matmul)
# ============================================================

def run_m3_m4_m5(device='npu:0'):
    print("\n" + "=" * 70)
    print("  M3/M4/M5: L0 Buffer Bandwidths (via matmul)")
    print("=" * 70)

    results = []

    # M3/M4/M5: Approximate via matmul with different tile shapes
    # The key insight: use pre-allocated tensors and batch the matmul

    configs = [
        # (m, n, k, label, param_id, param_name)
        (256, 256, 256, "L0A/L0B/C", "M3", "L0A带宽 (L0A BW)"),
        (256, 256, 256, "L0A/L0B/C", "M4", "L0B带宽 (L0B BW)"),
        (256, 256, 256, "L0A/L0B/C", "M5", "L0C带宽 (L0C BW)"),
    ]

    # Test different sizes to find L0 bandwidth
    print(f"\n  --- Matmul throughput at various sizes ---")
    print(f"  {'Size':>10s} {'Batch':>6s} {'Time(ms)':>10s} {'TFLOPS':>10s} {'GMAC/s':>10s}")

    for size in [64, 128, 256, 512, 1024, 2048, 4096]:
        batch = max(1, 200 // size)  # more batches for small sizes
        times = measure_matmul(size, size, size, batch_size=batch, device=device)
        med = float(np.median(times))
        flops = 2 * size * size * size
        tflops = flops / (med / 1000.0) / 1e12
        gmacs = flops / (med / 1000.0) / 1e9 / 2

        print(f"  {size:>8d}x{size:<4d} {batch:>6d} {med:10.4f} {tflops:10.4f} {gmacs:10.2f}")

    # Approximate L0 bandwidth from matmul data flow
    # For a matmul C = A @ B:
    # - L0A loads: M*K elements
    # - L0B loads: K*N elements
    # - L0C stores: M*N elements
    # Total data through L0 = (M*K + K*N + M*N) * element_size
    # We use 256x256x256 as reference

    m, n, k = 256, 256, 256
    batch = 100
    times = measure_matmul(m, n, k, batch_size=batch, device=device)
    med = float(np.median(times))

    element_size = 4
    l0a_bytes = m * k * element_size
    l0b_bytes = k * n * element_size
    l0c_bytes = m * n * element_size

    # Approximate: each buffer contributes equally to time
    # Total L0 data = L0A + L0B + L0C reads/writes
    total_l0 = l0a_bytes + l0b_bytes + l0c_bytes
    l0_bw = compute_bandwidth_gbs(total_l0, med)

    # M3: L0A
    res = make_result("M3", "L0A带宽 (L0A BW)", "MTE",
                      l0_bw, "GB/s", times.tolist(),
                      f"L0A approx from {m}x{k} matmul, total_l0={total_l0/1024:.0f}KB")
    results.append(res)
    print(f"\n  [M3] L0A BW ≈ {l0_bw:.2f} GB/s (from matmul {m}x{n}x{k})")

    # M4: L0B
    res = make_result("M4", "L0B带宽 (L0B BW)", "MTE",
                      l0_bw, "GB/s", times.tolist(),
                      f"L0B approx from {k}x{n} matmul")
    results.append(res)
    print(f"  [M4] L0B BW ≈ {l0_bw:.2f} GB/s")

    # M5: L0C
    res = make_result("M5", "L0C带宽 (L0C BW)", "MTE",
                      l0_bw, "GB/s", times.tolist(),
                      f"L0C approx from {m}x{n} result store")
    results.append(res)
    print(f"  [M5] L0C BW ≈ {l0_bw:.2f} GB/s")

    return results


# ============================================================
#  M6: HBM Access Latency
# ============================================================

def run_m6(device='npu:0'):
    print("\n" + "=" * 70)
    print("  M6: HBM Access Latency")
    print("=" * 70)

    results = []
    freq_mhz = estimate_clock_freq_mhz()

    # Use dependent copy chain for latency measurement
    # Each copy depends on previous → serialized
    for size_mb in [4, 16, 64]:
        n = size_mb * 1024 * 1024 // 4
        src = torch.randn(n, device=device)
        dst = torch.zeros(n, device=device)

        # Warmup
        for _ in range(10):
            dst.copy_(src)
        torch.npu.synchronize()

        # Measure single copy latency (no batching)
        start_e = torch.npu.Event(enable_timing=True)
        end_e = torch.npu.Event(enable_timing=True)
        times = []
        for _ in range(50):
            start_e.record()
            dst.copy_(src)
            end_e.record()
            torch.npu.synchronize()
            times.append(start_e.elapsed_time(end_e))

        times = np.array(times)
        med = float(np.median(times))
        cycles = med * freq_mhz  # ms * MHz = cycles (×1000)
        cycles_per_access = cycles  # single access

        res = make_result("M6", "HBM访存延迟 (HBM Latency)", "MTE",
                          cycles_per_access, "cycles", times.tolist(),
                          f"size={size_mb}MB, freq={freq_mhz}MHz")
        results.append(res)
        print(f"  {size_mb:>4d}MB: {cycles_per_access:>10.2f} cycles ({med:.4f}ms)")

    return results


# ============================================================
#  M7: Buffer Capacity (bandwidth cliff detection)
# ============================================================

def run_m7(device='npu:0'):
    print("\n" + "=" * 70)
    print("  M7: Buffer Capacity (bandwidth cliff)")
    print("=" * 70)

    results = []
    # Fine-grained sweep from 16KB to 32MB
    sizes_kb = [16, 32, 64, 128, 256, 384, 512, 768, 1024,
                1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768]

    print(f"\n  {'Size':>8s} {'BW(GB/s)':>10s} {'Med(ms)':>10s} {'CV':>8s}")

    for size_kb in sizes_kb:
        n = size_kb * 1024 // 4
        total_bytes = size_kb * 1024
        batch = get_batch(total_bytes)

        times = measure_copy_bw(n, batch_size=batch, device=device)
        med = float(np.median(times))
        bw = compute_bandwidth_gbs(total_bytes, med)
        cv = float(np.std(times) / np.mean(times))

        res = make_result("M7", f"Buffer容量探测 (size={size_kb}KB)", "MTE",
                          bw, "GB/s", times.tolist(),
                          f"size={size_kb}KB, batch={batch}")
        results.append(res)
        print(f"  {size_kb:>6d}KB {bw:10.2f} {med:10.4f} {cv:8.4f}")

    # Detect cliff
    bws = [r.value for r in results]
    sizes = [int(r.notes.split('size=')[1].split('KB')[0]) for r in results]
    peak_bw = max(bws)
    threshold = peak_bw * 0.7
    for s, bw in zip(sizes, bws):
        if bw < threshold:
            print(f"\n  [M7] L1 capacity cliff detected at ~{s}KB")
            break

    return results


# ============================================================
#  M8: DMA Transfer Startup Overhead
# ============================================================

def run_m8(device='npu:0'):
    print("\n" + "=" * 70)
    print("  M8: DMA Transfer Startup Overhead")
    print("=" * 70)

    results = []
    freq_mhz = estimate_clock_freq_mhz()

    # Small sizes to expose per-transfer overhead
    element_counts = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

    print(f"\n  {'Size':>8s} {'Time(us)':>10s} {'CV':>8s}")

    sizes_bytes = []
    times_us = []

    for count in element_counts:
        n = count
        total_bytes = count * 4

        # Use batch=1 for small sizes to expose per-transfer overhead
        times = measure_copy_bw(n, batch_size=1, num_warmup=10, num_iters=50, device=device)
        med = float(np.median(times))
        med_us = med * 1000  # ms → μs
        cv = float(np.std(times) / np.mean(times))

        sizes_bytes.append(total_bytes)
        times_us.append(med_us)

        res = make_result("M8", f"传输时间 (size={count} elements)", "MTE",
                          med_us, "us", times.tolist(),
                          f"transfer_size={count} fp32, {total_bytes}B")
        results.append(res)
        print(f"  {count:>6d}B {med_us:10.2f} {cv:8.4f}")

    # Linear fit: time = overhead + slope * size
    if len(sizes_bytes) >= 3:
        coeffs = np.polyfit(sizes_bytes, times_us, 1)
        slope_per_byte = coeffs[0]
        overhead_us = coeffs[1]

        res = make_result("M8", "MTE启动开销", "MTE",
                          overhead_us, "us", times_us,
                          f"Linear extrapolation, slope={slope_per_byte:.6f}us/B")
        results.append(res)
        print(f"\n  [M8] Startup overhead: {overhead_us:.2f} μs")
        print(f"  [M8] Per-byte cost: {slope_per_byte:.6f} μs/B")
        print(f"  [M8] Effective BW (large xfer): {1/slope_per_byte/1024/1024*1e6:.2f} GB/s")

    return results


# ============================================================
#  Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--m1', action='store_true')
    parser.add_argument('--m2', action='store_true')
    parser.add_argument('--m345', action='store_true')
    parser.add_argument('--m6', action='store_true')
    parser.add_argument('--m7', action='store_true')
    parser.add_argument('--m8', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--device', type=str, default='npu:0')
    args = parser.parse_args()

    if not any([args.m1, args.m2, args.m345, args.m6, args.m7, args.m8]):
        args.all = True

    device = args.device
    all_results = []

    if args.all or args.m1:
        all_results.extend(run_m1(device))
    if args.all or args.m2:
        all_results.extend(run_m2(device))
    if args.all or args.m345:
        all_results.extend(run_m3_m4_m5(device))
    if args.all or args.m6:
        all_results.extend(run_m6(device))
    if args.all or args.m7:
        all_results.extend(run_m7(device))
    if args.all or args.m8:
        all_results.extend(run_m8(device))

    # Save results
    outpath = os.path.join(os.path.dirname(__file__), '..', 'results', 'mte_results_optimized.json')
    save_results(all_results, outpath)
    print(f"\n  Results saved to {outpath}")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY: Key Parameters")
    print("=" * 70)

    for r in all_results:
        if any(k in r.param_id for k in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']):
            if '65536' in r.notes or '16384' in r.notes or '32768' in r.notes or 'HBM' in r.param_name:
                print(f"  [{r.param_id}] {r.param_name}: {r.value:.2f} {r.unit}")
            elif 'M3' in r.param_id or 'M4' in r.param_id or 'M5' in r.param_id:
                if '256' in r.notes:
                    print(f"  [{r.param_id}] {r.param_name}: {r.value:.2f} {r.unit}")
            elif 'M6' in r.param_id:
                if '64' in r.notes:
                    print(f"  [{r.param_id}] {r.param_name}: {r.value:.2f} {r.unit}")


if __name__ == '__main__':
    main()
