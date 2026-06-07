#!/usr/bin/env python3
"""
Run all optimized benchmarks and generate comprehensive results.
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from datetime import datetime

# Import all benchmark modules
from common.benchmark import BenchResult, save_results, print_result
from common.utils import compute_bandwidth_gbs, estimate_clock_freq_mhz, ms_to_cycles

try:
    import torch
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


def run_scalar_benchmarks():
    """Run S1-S3 scalar benchmarks."""
    print("\n" + "=" * 70)
    print("  SCALAR UNIT BENCHMARKS (S1-S3)")
    print("=" * 70)

    from scalar.s1_latency import ScalarArithmeticLatencyBench as ScalarLatencyBench
    from scalar.s2_throughput import ScalarThroughputBench
    from scalar.s3_mem_latency import ScalarMemAccessLatencyBench as ScalarMemLatencyBench

    results = []

    # S1: Scalar arithmetic latency
    print("\n--- S1: Scalar Arithmetic Latency ---")
    s1 = ScalarLatencyBench()
    s1_results = s1.measure_latency()
    results.extend(s1_results)

    # S2: Scalar throughput
    print("\n--- S2: Scalar Unit Throughput ---")
    s2 = ScalarThroughputBench()
    s2_results = s2.measure_throughput()
    results.extend(s2_results)

    # S3: Scalar memory latency
    print("\n--- S3: Scalar Memory Access Latency ---")
    s3 = ScalarMemLatencyBench()
    s3_results = s3.measure_latency()
    results.extend(s3_results)

    return results


def run_vector_benchmarks():
    """Run V1-V5 vector benchmarks."""
    print("\n" + "=" * 70)
    print("  VECTOR UNIT BENCHMARKS (V1-V5)")
    print("=" * 70)

    from vector.v1_add_latency import VectorAddLatencyBench
    from vector.v2_mul_latency import VectorMulLatencyBench
    from vector.v3_throughput import VectorThroughputBench
    from vector.v4_pipeline_depth import VectorPipelineDepthBench
    from vector.v5_reg_latency import VectorRegAccessLatencyBench as VectorRegLatencyBench

    results = []

    # V1: Vector add latency
    print("\n--- V1: FP32 Vector Add Latency ---")
    v1 = VectorAddLatencyBench()
    v1_results = v1.measure_latency()
    results.extend(v1_results)

    # V2: Vector mul latency
    print("\n--- V2: FP32 Vector Mul Latency ---")
    v2 = VectorMulLatencyBench()
    v2_results = v2.measure_latency()
    results.extend(v2_results)

    # V3: Vector throughput
    print("\n--- V3: Vector Unit Throughput ---")
    v3 = VectorThroughputBench()
    v3_results = v3.measure_throughput()
    results.extend(v3_results)

    # V4: Pipeline depth
    print("\n--- V4: Vector Pipeline Depth ---")
    v4 = VectorPipelineDepthBench()
    v4_results = v4.measure_pipeline_depth()
    results.extend(v4_results)

    # V5: Register latency
    print("\n--- V5: Vector Register Access Latency ---")
    v5 = VectorRegLatencyBench()
    v5_results = v5.measure_latency()
    results.extend(v5_results)

    return results


def run_cube_benchmarks():
    """Run C1-C5 cube benchmarks."""
    print("\n" + "=" * 70)
    print("  CUBE UNIT BENCHMARKS (C1-C5)")
    print("=" * 70)

    from cube.c1_tile_latency import SingleTileLatencyBench as TileLatencyBench
    from cube.c2_throughput import MatMulThroughputBench
    from cube.c3_pipeline_depth import CubePipelineDepthBench
    from cube.c4_l0_latency import CubeBufferLatencyBench as L0LatencyBench
    from cube.c5_scaling import MatMulScalingBench

    results = []

    # C1: Tile latency
    print("\n--- C1: MatMul Latency (Single Tile) ---")
    c1 = TileLatencyBench()
    c1_results = c1.measure_latency()
    results.extend(c1_results)

    # C2: MatMul throughput
    print("\n--- C2: MatMul Throughput ---")
    c2 = MatMulThroughputBench()
    c2_results = c2.measure_throughput()
    results.extend(c2_results)

    # C3: Pipeline depth
    print("\n--- C3: Cube Pipeline Depth ---")
    c3 = CubePipelineDepthBench()
    c3_results = c3.measure_pipeline_depth()
    results.extend(c3_results)

    # C4: L0 latency
    print("\n--- C4: L0 Access Latency ---")
    c4 = L0LatencyBench()
    c4_results = c4.measure_latency()
    results.extend(c4_results)

    # C5: Scaling
    print("\n--- C5: MatMul Scaling ---")
    c5 = MatMulScalingBench()
    c5_results = c5.measure_scaling()
    results.extend(c5_results)

    return results


def run_mte_benchmarks():
    """Run M1-M8 MTE benchmarks."""
    print("\n" + "=" * 70)
    print("  MTE BENCHMARKS (M1-M8)")
    print("=" * 70)

    from mte.m1_l1_read_bw import L1ReadBandwidthBench
    from mte.m2_l1_write_bw import L1WriteBandwidthBench
    from mte.m3_m4_m5_l0_bw import L0BandwidthBench
    from mte.m6_hbm_latency import HBMMemLatencyBench
    from mte.m7_buffer_capacity import BufferCapacityBench
    from mte.m8_startup_overhead import TransferStartupOverheadBench

    results = []

    # M1: L1 read BW
    print("\n--- M1: L1 Buffer Read Bandwidth ---")
    m1 = L1ReadBandwidthBench()
    m1_results = m1.measure_bandwidth()
    results.extend(m1_results)

    # M2: L1 write BW
    print("\n--- M2: L1 Buffer Write Bandwidth ---")
    m2 = L1WriteBandwidthBench()
    m2_results = m2.measure_bandwidth()
    results.extend(m2_results)

    # M3/M4/M5: L0 BW
    print("\n--- M3/M4/M5: L0 Buffer Bandwidths ---")
    m345 = L0BandwidthBench()
    m345_results = m345.measure_all()
    results.extend(m345_results)

    # M6: HBM latency
    print("\n--- M6: HBM Access Latency ---")
    m6 = HBMMemLatencyBench()
    m6_results = m6.measure_latency()
    results.extend(m6_results)

    # M7: Buffer capacity
    print("\n--- M7: Buffer Capacity ---")
    m7 = BufferCapacityBench()
    m7_results = m7.measure_l1_capacity()
    results.extend(m7_results)

    # M8: Startup overhead
    print("\n--- M8: DMA Startup Overhead ---")
    m8 = TransferStartupOverheadBench()
    m8_results = m8.measure_startup_overhead()
    results.extend(m8_results)

    return results


def extract_key_params(all_results):
    """Extract key parameters from all results."""
    params = {}

    for r in all_results:
        pid = r.param_id

        # S1: scalar latency (take median of longest chain)
        if pid == 'S1' and 'chain_length=1000' in r.notes:
            params['S1'] = r.value

        # S2: scalar throughput (take best)
        if pid == 'S2':
            if 'S2' not in params or r.value > params.get('S2', 0):
                params['S2'] = r.value

        # S3: scalar memory latency
        if pid == 'S3' and 'num_accesses=1000' in r.notes:
            params['S3'] = r.value

        # V1: vector add latency
        if pid == 'V1' and 'vec_size=256' in r.notes and 'chain_len=500' in r.notes:
            params['V1'] = r.value

        # V2: vector mul latency
        if pid == 'V2' and 'vec_size=256' in r.notes and 'chain_len=500' in r.notes:
            params['V2'] = r.value

        # V3: vector throughput (best)
        if pid == 'V3':
            if 'V3' not in params or r.value > params.get('V3', 0):
                params['V3'] = r.value

        # V5: register latency
        if pid == 'V5_READ':
            params['V5_READ'] = r.value
        if pid == 'V5_WRITE':
            params['V5_WRITE'] = r.value

        # C1: tile latency
        if pid == 'C1' and 'tile=4x4' in r.notes:
            params['C1'] = r.value

        # C2: matmul throughput (best TFLOPS)
        if pid == 'C2':
            if 'C2' not in params or r.value > params.get('C2', 0):
                params['C2'] = r.value

        # C4: L0 latency
        if pid == 'C4_READ':
            params['C4_READ'] = r.value
        if pid == 'C4_WRITE':
            params['C4_WRITE'] = r.value

        # M1: L1 read BW (peak)
        if pid == 'M1' and 'L1' in r.notes:
            if 'M1' not in params or r.value > params.get('M1', 0):
                params['M1'] = r.value

        # M2: L1 write BW (peak)
        if pid == 'M2' and 'L1' in r.notes:
            if 'M2' not in params or r.value > params.get('M2', 0):
                params['M2'] = r.value

        # M3/M4/M5: L0 BW
        if pid == 'M3':
            params['M3'] = r.value
        if pid == 'M4':
            params['M4'] = r.value
        if pid == 'M5':
            params['M5'] = r.value

        # M6: HBM latency (largest array)
        if pid == 'M6' and '64' in r.notes:
            params['M6'] = r.value

        # M8: startup overhead
        if pid == 'M8' and r.param_name == 'MTE启动开销':
            params['M8'] = r.value

    return params


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scalar', action='store_true')
    parser.add_argument('--vector', action='store_true')
    parser.add_argument('--cube', action='store_true')
    parser.add_argument('--mte', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if not any([args.scalar, args.vector, args.cube, args.mte]):
        args.all = True

    all_results = []

    if args.all or args.scalar:
        all_results.extend(run_scalar_benchmarks())
    if args.all or args.vector:
        all_results.extend(run_vector_benchmarks())
    if args.all or args.cube:
        all_results.extend(run_cube_benchmarks())
    if args.all or args.mte:
        all_results.extend(run_mte_benchmarks())

    # Save all results
    outdir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(outdir, exist_ok=True)

    # Save individual category results
    scalar_results = [r for r in all_results if 'Scalar' in r.category]
    vector_results = [r for r in all_results if 'Vector' in r.category]
    cube_results = [r for r in all_results if 'Cube' in r.category]
    mte_results = [r for r in all_results if 'MTE' in r.category]

    save_results(scalar_results, os.path.join(outdir, 'scalar_results.json'))
    save_results(vector_results, os.path.join(outdir, 'vector_results.json'))
    save_results(cube_results, os.path.join(outdir, 'cube_results.json'))
    save_results(mte_results, os.path.join(outdir, 'mte_results.json'))

    # Save all results
    save_results(all_results, os.path.join(outdir, 'all_results.json'))

    # Extract and save key parameters
    params = extract_key_params(all_results)
    with open(os.path.join(outdir, 'extracted_params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    print(f"\n[INFO] Extracted {len(params)} key parameters")

    # Print summary
    print("\n" + "=" * 70)
    print("  EXTRACTED KEY PARAMETERS")
    print("=" * 70)
    for k, v in sorted(params.items()):
        unit = ''
        if k.startswith('S') and '1' in k: unit = 'cycles'
        elif k == 'S2': unit = 'ops/cycle'
        elif k == 'S3': unit = 'cycles'
        elif k.startswith('V') and ('latency' in k.lower() or k in ['V1','V2']): unit = 'cycles'
        elif k == 'V3': unit = 'ops/cycle'
        elif k.startswith('V5'): unit = 'cycles'
        elif k.startswith('C1'): unit = 'cycles'
        elif k == 'C2': unit = 'TFLOPS'
        elif k.startswith('C4'): unit = 'cycles'
        elif k.startswith('M') and k[-1].isdigit() and int(k[1:]) <= 5: unit = 'GB/s'
        elif k == 'M6': unit = 'cycles'
        elif k == 'M8': unit = 'cycles'
        print(f"  {k}: {v:.4f} {unit}")


if __name__ == '__main__':
    main()
