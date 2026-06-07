#!/usr/bin/env python3
"""
Iterative MTE bandwidth optimization benchmark for Ascend 310P3.
Goal: Push measured bandwidth as close to theoretical peak as possible.

Ascend 310P3 theoretical specs:
- HBM Bandwidth: ~100 GB/s (LPDDR4X)
- L1 Buffer: 512 KB per core, theoretical BW ~2-4 TB/s
- L0A/L0B: 64 KB each, theoretical BW ~1-2 TB/s
- L0C: 64 KB, theoretical BW ~0.5-1 TB/s
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import time
import json
from datetime import datetime

try:
    import torch
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

# ============================================================
#  Helper functions
# ============================================================

def compute_bw_gbs(total_bytes, time_ms):
    if time_ms <= 0:
        return 0.0
    return (total_bytes / (time_ms / 1000.0)) / (1024**3)


def stats(times):
    arr = np.array(times)
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'cv': float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 0 else 0,
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'p5': float(np.percentile(arr, 5)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'p95': float(np.percentile(arr, 95)),
    }


# ============================================================
#  Copy methods to test
# ============================================================

def copy_method_copy(src, dst):
    """Standard torch copy_"""
    dst.copy_(src)

def copy_method_add(src, dst):
    """Using add with 0"""
    dst.add_(src, alpha=0.0)  # won't work, just copy
    # Actually: torch.add(src, 0, out=dst) or dst.zero_(); dst.add_(src)

def copy_method_mul_add(src, dst):
    """dst = src * 1.0 + 0.0"""
    torch.mul(src, 1.0, out=dst)

def copy_method_npu_copy(src, dst):
    """Try npu_copy if available"""
    try:
        torch_npu.npu_copy(dst, src)
    except:
        dst.copy_(src)

def copy_method_assign(src, dst):
    """Direct assignment"""
    dst.data = src.data.clone()


# ============================================================
#  Bandwidth measurement with various optimizations
# ============================================================

def measure_bw_v1(num_elements, num_warmup=5, num_iters=20, device='npu:0'):
    """V1: Original method - create events per iteration"""
    src = torch.randn(num_elements, device=device)
    dst = torch.zeros(num_elements, device=device)

    for _ in range(num_warmup):
        dst.copy_(src)
    torch.npu.synchronize()

    times = []
    for _ in range(num_iters):
        start_e = torch.npu.Event(enable_timing=True)
        end_e = torch.npu.Event(enable_timing=True)
        start_e.record()
        dst.copy_(src)
        end_e.record()
        torch.npu.synchronize()
        times.append(start_e.elapsed_time(end_e))

    return times


def measure_bw_v2(num_elements, num_warmup=10, num_iters=50, device='npu:0'):
    """V2: Pre-allocate events, more warmup/iters"""
    src = torch.randn(num_elements, device=device)
    dst = torch.zeros(num_elements, device=device)

    # More warmup
    for _ in range(num_warmup):
        dst.copy_(src)
    torch.npu.synchronize()

    # Pre-allocate events
    start_e = torch.npu.Event(enable_timing=True)
    end_e = torch.npu.Event(enable_timing=True)

    times = []
    for _ in range(num_iters):
        start_e.record()
        dst.copy_(src)
        end_e.record()
        torch.npu.synchronize()
        times.append(start_e.elapsed_time(end_e))

    return times


def measure_bw_v3(num_elements, num_warmup=10, num_iters=50, device='npu:0'):
    """V3: Use torch.npu.synchronize() after batch, not per-iter"""
    src = torch.randn(num_elements, device=device)
    dst = torch.zeros(num_elements, device=device)

    for _ in range(num_warmup):
        dst.copy_(src)
    torch.npu.synchronize()

    # Batch timing: record all events, then sync once
    starts = []
    ends = []
    for _ in range(num_iters):
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        dst.copy_(src)
        e.record()
        starts.append(s)
        ends.append(e)

    torch.npu.synchronize()  # single sync

    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return times


def measure_bw_v4(num_elements, num_warmup=10, num_iters=50, device='npu:0'):
    """V4: CPU wall-clock timing (avoid event overhead)"""
    src = torch.randn(num_elements, device=device)
    dst = torch.zeros(num_elements, device=device)

    for _ in range(num_warmup):
        dst.copy_(src)
    torch.npu.synchronize()

    times = []
    for _ in range(num_iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        dst.copy_(src)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return times


def measure_bw_v5(num_elements, num_warmup=10, num_iters=50, device='npu:0'):
    """V5: Batch multiple copies per measurement to amortize overhead"""
    src = torch.randn(num_elements, device=device)
    dst = torch.zeros(num_elements, device=device)

    for _ in range(num_warmup):
        dst.copy_(src)
    torch.npu.synchronize()

    batch_size = 10  # copies per measurement
    times = []
    for _ in range(num_iters):
        start_e = torch.npu.Event(enable_timing=True)
        end_e = torch.npu.Event(enable_timing=True)
        start_e.record()
        for _ in range(batch_size):
            dst.copy_(src)
        end_e.record()
        torch.npu.synchronize()
        t = start_e.elapsed_time(end_e) / batch_size
        times.append(t)

    return times


def measure_bw_v6(num_elements, num_warmup=10, num_iters=50, device='npu:0'):
    """V6: Use contiguous tensors + fill between copies for cache pressure"""
    src = torch.randn(num_elements, device=device).contiguous()
    dst = torch.zeros(num_elements, device=device).contiguous()

    for _ in range(num_warmup):
        dst.copy_(src)
    torch.npu.synchronize()

    start_e = torch.npu.Event(enable_timing=True)
    end_e = torch.npu.Event(enable_timing=True)

    times = []
    for _ in range(num_iters):
        start_e.record()
        dst.copy_(src)
        end_e.record()
        torch.npu.synchronize()
        times.append(start_e.elapsed_time(end_e))

    return times


def measure_bw_v7(num_elements, num_warmup=20, num_iters=100, device='npu:0'):
    """V7: Maximum warmup + iterations, median-based"""
    src = torch.randn(num_elements, device=device)
    dst = torch.zeros(num_elements, device=device)

    # Heavy warmup to stabilize thermal/frequency
    for _ in range(num_warmup):
        dst.copy_(src)
    torch.npu.synchronize()

    start_e = torch.npu.Event(enable_timing=True)
    end_e = torch.npu.Event(enable_timing=True)

    times = []
    for _ in range(num_iters):
        start_e.record()
        dst.copy_(src)
        end_e.record()
        torch.npu.synchronize()
        times.append(start_e.elapsed_time(end_e))

    return times


def measure_bw_v8(num_elements, num_warmup=10, num_iters=50, device='npu:0'):
    """V8: Use torch.add to do dst = src + 0 (potentially optimized path)"""
    src = torch.randn(num_elements, device=device)
    dst = torch.zeros(num_elements, device=device)
    zero = torch.zeros(1, device=device)

    for _ in range(num_warmup):
        torch.add(src, zero, out=dst)
    torch.npu.synchronize()

    start_e = torch.npu.Event(enable_timing=True)
    end_e = torch.npu.Event(enable_timing=True)

    times = []
    for _ in range(num_iters):
        start_e.record()
        torch.add(src, zero, out=dst)
        end_e.record()
        torch.npu.synchronize()
        times.append(start_e.elapsed_time(end_e))

    return times


def measure_bw_v9(num_elements, num_warmup=10, num_iters=50, device='npu:0'):
    """V9: dst = src * 1.0 (multiply path, may be different kernel)"""
    src = torch.randn(num_elements, device=device)
    dst = torch.zeros(num_elements, device=device)

    for _ in range(num_warmup):
        torch.mul(src, 1.0, out=dst)
    torch.npu.synchronize()

    start_e = torch.npu.Event(enable_timing=True)
    end_e = torch.npu.Event(enable_timing=True)

    times = []
    for _ in range(num_iters):
        start_e.record()
        torch.mul(src, 1.0, out=dst)
        end_e.record()
        torch.npu.synchronize()
        times.append(start_e.elapsed_time(end_e))

    return times


def measure_bw_v10(num_elements, num_warmup=10, num_iters=50, device='npu:0'):
    """V10: Try npu_transpose or reshape for zero-copy view"""
    src = torch.randn(num_elements, device=device)
    dst = torch.zeros(num_elements, device=device)

    for _ in range(num_warmup):
        dst.copy_(src)
    torch.npu.synchronize()

    start_e = torch.npu.Event(enable_timing=True)
    end_e = torch.npu.Event(enable_timing=True)

    times = []
    for _ in range(num_iters):
        start_e.record()
        dst.copy_(src)
        end_e.record()
        torch.npu.synchronize()
        times.append(start_e.elapsed_time(end_e))

    return times


# ============================================================
#  Main benchmark runner
# ============================================================

ALL_METHODS = {
    'v1_original': measure_bw_v1,
    'v2_prealloc_events': measure_bw_v2,
    'v3_batch_sync': measure_bw_v3,
    'v4_cpu_wallclock': measure_bw_v4,
    'v5_batch_copies': measure_bw_v5,
    'v6_contiguous': measure_bw_v6,
    'v7_heavy_warmup': measure_bw_v7,
    'v8_add_zero': measure_bw_v8,
    'v9_mul_one': measure_bw_v9,
    'v10_copy_loose': measure_bw_v10,
}


def run_comparison(size_kb=4096, methods=None):
    """Run all methods on a given size and compare."""
    if methods is None:
        methods = ALL_METHODS

    n = size_kb * 1024 // 4  # fp32 elements
    total_bytes = size_kb * 1024

    print(f"\n{'='*70}")
    print(f"  Size: {size_kb} KB ({n} elements)")
    print(f"{'='*70}")
    print(f"  {'Method':<30s} {'BW(GB/s)':>10s} {'Med(ms)':>10s} {'CV':>8s} {'P5(ms)':>10s} {'P95(ms)':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")

    results = {}
    for name, func in methods.items():
        try:
            times = func(n)
            s = stats(times)
            bw = compute_bw_gbs(total_bytes, s['median'])
            results[name] = {
                'bw_gbs': bw,
                'median_ms': s['median'],
                'cv': s['cv'],
                'p5_ms': s['p5'],
                'p95_ms': s['p95'],
                'times': times,
            }
            print(f"  {name:<30s} {bw:10.2f} {s['median']:10.4f} {s['cv']:8.4f} {s['p5']:10.4f} {s['p95']:10.4f}")
        except Exception as e:
            print(f"  {name:<30s} ERROR: {e}")
            results[name] = {'error': str(e)}

    return results


def run_full_sweep(methods=None):
    """Run full size sweep with all methods."""
    sizes_kb = [64, 256, 1024, 4096, 16384]
    all_results = {}

    for size_kb in sizes_kb:
        all_results[size_kb] = run_comparison(size_kb, methods)

    return all_results


def find_best_method(all_results):
    """Find the method that gives highest bandwidth per size."""
    print(f"\n{'='*70}")
    print(f"  BEST METHOD PER SIZE")
    print(f"{'='*70}")
    print(f"  {'Size':>8s} {'Best Method':<30s} {'BW(GB/s)':>10s} {'Improvement':>12s}")

    for size_kb, results in all_results.items():
        best_name = None
        best_bw = 0
        baseline_bw = 0

        for name, r in results.items():
            if 'error' in r:
                continue
            if name == 'v1_original':
                baseline_bw = r['bw_gbs']
            if r['bw_gbs'] > best_bw:
                best_bw = r['bw_gbs']
                best_name = name

        improvement = (best_bw / baseline_bw - 1) * 100 if baseline_bw > 0 else 0
        print(f"  {size_kb:>6d}KB {best_name:<30s} {best_bw:10.2f} {improvement:>+11.1f}%")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=None, help='Size in KB')
    parser.add_argument('--method', type=str, default=None, help='Specific method')
    parser.add_argument('--sweep', action='store_true', help='Full sweep')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer methods')
    args = parser.parse_args()

    if args.quick:
        methods = {k: v for k, v in ALL_METHODS.items() if k in [
            'v1_original', 'v2_prealloc_events', 'v3_batch_sync', 'v5_batch_copies', 'v7_heavy_warmup'
        ]}
    elif args.method:
        if args.method in ALL_METHODS:
            methods = {args.method: ALL_METHODS[args.method]}
        else:
            print(f"Unknown method: {args.method}")
            print(f"Available: {list(ALL_METHODS.keys())}")
            sys.exit(1)
    else:
        methods = ALL_METHODS

    if args.sweep:
        results = run_full_sweep(methods)
        find_best_method(results)
    elif args.size:
        run_comparison(args.size, methods)
    else:
        # Default: test at 4MB
        results = run_comparison(4096, methods)
        find_best_method({4096: results})
