#!/usr/bin/env python3
"""
Trace-Based Ground Truth Extraction
从 simulator_trace/trace.json 中提取 21 个参数的实际值（ground truth）。

使用方法:
  python extract_ground_truth.py [--trace trace.json] [--output ground_truth.json]

Trace JSON 格式说明:
trace.json 是一个操作列表，每个操作记录包含:
  - op_name: 操作名称 (如 "add", "matmul", "copy")
  - start_time: 开始时间 (ns)
  - end_time: 结束时间 (ns)
  - latency: 延迟 (cycles)
  - input_size: 输入数据大小
  - unit: 执行的硬件单元 (scalar/vector/cube/mte)
  - details: 额外细节
"""

import json
import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class TraceOp:
    """A single operation from the trace."""
    op_name: str
    start_time: float  # ns
    end_time: float    # ns
    latency: float     # cycles
    unit: str          # scalar/vector/cube/mte
    input_size: Tuple[int, ...] = ()
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ns(self) -> float:
        return self.end_time - self.start_time


def load_trace(trace_path: str) -> List[Dict]:
    """Load trace from JSON file."""
    with open(trace_path, 'r') as f:
        data = json.load(f)

    # Trace may be a list or a dict with 'ops' key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'ops' in data:
        return data['ops']
    else:
        raise ValueError(f"Unexpected trace format: {type(data)}")


def parse_ops(raw_ops: List[Dict]) -> List[TraceOp]:
    """Parse raw trace entries into TraceOp objects."""
    ops = []
    for raw in raw_ops:
        op = TraceOp(
            op_name=raw.get('op_name', raw.get('name', 'unknown')),
            start_time=raw.get('start_time', raw.get('start', 0)),
            end_time=raw.get('end_time', raw.get('end', 0)),
            latency=raw.get('latency', raw.get('cycles', 0)),
            unit=raw.get('unit', raw.get('exec_unit', 'unknown')),
            input_size=tuple(raw.get('input_size', raw.get('shape', ()))),
            details={k: v for k, v in raw.items()
                     if k not in ['op_name', 'name', 'start_time', 'start',
                                  'end_time', 'end', 'latency', 'cycles',
                                  'unit', 'exec_unit', 'input_size', 'shape']}
        )
        ops.append(op)
    return ops


class GroundTruthExtractor:
    """
    从 trace 数据中提取各微架构参数的实际值。

    核心思想：
    - 对于延迟类参数：找到对应操作的 trace 记录，取延迟的平均值/中位数
    - 对于吞吐率参数：找到批量操作的 trace，计算总操作数/总时间
    - 对于带宽参数：找到数据搬运的 trace，计算数据量/时间
    - 对于容量参数：找到性能悬崖点
    """

    def __init__(self, ops: List[TraceOp], clock_freq_mhz: float = 1000.0):
        self.ops = ops
        self.clock_freq_mhz = clock_freq_mhz
        self.ns_per_cycle = 1000.0 / clock_freq_mhz  # ns

        # Group ops by category
        self.scalar_ops = [op for op in ops if 'scalar' in op.unit.lower()]
        self.vector_ops = [op for op in ops if 'vector' in op.unit.lower()]
        self.cube_ops = [op for op in ops if 'cube' in op.unit.lower()]
        self.mte_ops = [op for op in ops if 'mte' in op.unit.lower() or
                        'memory' in op.unit.lower() or
                        'dma' in op.unit.lower()]

    def extract_all(self) -> Dict[str, Any]:
        """提取所有 21 个参数的实际值。"""
        results = {}

        # Scalar Unit
        results.update(self._extract_scalar_params())

        # Vector Unit
        results.update(self._extract_vector_params())

        # Cube Unit
        results.update(self._extract_cube_params())

        # MTE
        results.update(self._extract_mte_params())

        return results

    def _ns_to_cycles(self, ns: float) -> float:
        return ns / self.ns_per_cycle

    def _extract_scalar_params(self) -> Dict[str, Any]:
        """提取 S1-S3 的实际值。"""
        params = {}

        # S1: 标量算术指令延迟
        # 查找最小的标量加法操作，取其中位数延迟
        add_ops = [op for op in self.scalar_ops
                   if 'add' in op.op_name.lower() or 'scalar_add' in op.op_name.lower()]
        if add_ops:
            latencies = [op.latency for op in add_ops]
            params['S1'] = {
                'value': float(np.median(latencies)),
                'unit': 'cycles',
                'method': 'median latency of scalar add ops from trace',
                'num_samples': len(latencies),
                'std': float(np.std(latencies)) if len(latencies) > 1 else 0,
                'p25': float(np.percentile(latencies, 25)),
                'p75': float(np.percentile(latencies, 75)),
            }

        # S2: 标量单元吞吐率
        # 计算单位时间内完成的标量操作数
        # throughput = total_ops / total_time_in_cycles * frequency
        if self.scalar_ops:
            total_ops = len(self.scalar_ops)
            total_time_ns = sum(
                op.end_time - op.start_time for op in self.scalar_ops
            )
            if total_time_ns > 0:
                total_cycles = self._ns_to_cycles(total_time_ns)
                ops_per_cycle = total_ops / total_cycles if total_cycles > 0 else 0
                params['S2'] = {
                    'value': float(ops_per_cycle),
                    'unit': 'ops/cycle',
                    'method': 'total scalar ops / total cycles',
                    'num_ops': total_ops,
                    'total_cycles': float(total_cycles),
                }

        # S3: 标量访存延迟
        load_ops = [op for op in self.scalar_ops
                    if 'load' in op.op_name.lower() or 'store' in op.op_name.lower()]
        if load_ops:
            latencies = [op.latency for op in load_ops]
            params['S3'] = {
                'value': float(np.median(latencies)),
                'unit': 'cycles',
                'method': 'median latency of scalar load/store ops from trace',
                'num_samples': len(latencies),
                'std': float(np.std(latencies)) if len(latencies) > 1 else 0,
            }

        return params

    def _extract_vector_params(self) -> Dict[str, Any]:
        """提取 V1-V5 的实际值。"""
        params = {}

        # V1: FP32 向量加法延迟
        vec_add_ops = [op for op in self.vector_ops
                       if 'add' in op.op_name.lower() and 'vec' in op.op_name.lower()]
        if vec_add_ops:
            latencies = [op.latency for op in vec_add_ops]
            params['V1'] = {
                'value': float(np.median(latencies)),
                'unit': 'cycles',
                'method': 'median latency of vector add ops',
                'num_samples': len(latencies),
                'std': float(np.std(latencies)) if len(latencies) > 1 else 0,
            }

        # V2: FP32 向量乘法延迟
        vec_mul_ops = [op for op in self.vector_ops
                       if 'mul' in op.op_name.lower() and 'vec' in op.op_name.lower()]
        if vec_mul_ops:
            latencies = [op.latency for op in vec_mul_ops]
            params['V2'] = {
                'value': float(np.median(latencies)),
                'unit': 'cycles',
                'method': 'median latency of vector mul ops',
                'num_samples': len(latencies),
                'std': float(np.std(latencies)) if len(latencies) > 1 else 0,
            }

        # V3: 向量单元吞吐率
        if self.vector_ops:
            total_ops = len(self.vector_ops)
            total_time_ns = sum(op.end_time - op.start_time for op in self.vector_ops)
            if total_time_ns > 0:
                total_cycles = self._ns_to_cycles(total_time_ns)
                ops_per_cycle = total_ops / total_cycles if total_cycles > 0 else 0
                params['V3'] = {
                    'value': float(ops_per_cycle),
                    'unit': 'ops/cycle',
                    'method': 'total vector ops / total cycles',
                    'num_ops': total_ops,
                }

        # V4: 向量单元流水线深度
        # 从 trace 中间接推断：最大同时执行的向量操作数
        # 方法：扫描时间轴，统计任意时刻活跃的向量操作数
        if self.vector_ops:
            max_concurrent = self._compute_max_concurrent(self.vector_ops)
            params['V4'] = {
                'value': max_concurrent,
                'unit': '条',
                'method': 'max concurrent vector ops at any time point',
            }

        # V5: 向量寄存器访问延迟
        # 从第一条向量操作到第二条之间的间隔（无依赖时最小间隔）
        if len(self.vector_ops) >= 2:
            ops_sorted = sorted(self.vector_ops, key=lambda op: op.start_time)
            intervals = []
            for i in range(len(ops_sorted) - 1):
                gap = ops_sorted[i+1].start_time - ops_sorted[i].end_time
                if gap > 0:
                    intervals.append(self._ns_to_cycles(gap))
            if intervals:
                params['V5'] = {
                    'value': float(np.median(intervals)),
                    'unit': 'cycles',
                    'method': 'median gap between consecutive vector ops',
                    'note': 'Represents register write-back + bypass latency',
                }

        return params

    def _extract_cube_params(self) -> Dict[str, Any]:
        """提取 C1-C5 的实际值。"""
        params = {}

        # C1: 单 tile 矩阵乘延迟 (找最小 matmul)
        matmul_ops = [op for op in self.cube_ops
                      if 'matmul' in op.op_name.lower() or 'gemm' in op.op_name.lower()]
        if matmul_ops:
            # 找 input_size 最小的操作
            matmul_ops_sorted = sorted(matmul_ops,
                                       key=lambda op: (op.input_size[0] if op.input_size else 999999))
            smallest = matmul_ops_sorted[:min(10, len(matmul_ops_sorted))]
            latencies = [op.latency for op in smallest]
            params['C1'] = {
                'value': float(np.median(latencies)),
                'unit': 'cycles',
                'method': 'median latency of smallest matmul ops',
                'tile_sizes': [op.input_size for op in smallest],
            }

            # C2: 矩阵乘吞吐率
            # 对所有 matmul ops: total_MACs / total_time
            total_macs = 0
            for op in matmul_ops:
                if len(op.input_size) >= 3:
                    m, n, k = op.input_size[:3]
                    total_macs += 2 * m * n * k  # 1 MAC = 2 FLOPs
            total_time_ns = sum(op.end_time - op.start_time for op in matmul_ops)
            if total_time_ns > 0:
                mac_per_s = total_macs / (total_time_ns * 1e-9)
                tflops = (mac_per_s * 2) / 1e12
                params['C2'] = {
                    'value': float(tflops),
                    'unit': 'TFLOPS',
                    'method': 'total MAC operations / total time',
                    'total_macs': total_macs,
                }

            # C3: 矩阵乘流水线深度
            max_concurrent = self._compute_max_concurrent(matmul_ops)
            params['C3'] = {
                'value': max_concurrent,
                'unit': '条',
                'method': 'max concurrent matmul ops at any time point',
            }

            # C4: L0A/L0B/L0C 访问延迟
            # 从 matmul 的 data loading phase 推断
            # L0 load 延迟 = 最小 matmul 操作中 data load 部分
            load_phases = []
            for op in matmul_ops[:50]:
                if 'load_latency' in op.details:
                    load_phases.append(op.details['load_latency'])
                elif 'l0_read' in op.details:
                    load_phases.append(op.details['l0_read'])
            if load_phases:
                params['C4'] = {
                    'value': float(np.median(load_phases)),
                    'unit': 'cycles',
                    'method': 'median L0 data load phase latency',
                    'note': 'Combined L0A/L0B write + L0C read',
                }

            # C5: 不同规模矩阵乘延迟缩放关系
            scaling_data = []
            for op in matmul_ops:
                if len(op.input_size) >= 3:
                    m, n, k = op.input_size[:3]
                    scaling_data.append({
                        'm': m, 'n': n, 'k': k,
                        'latency': op.latency,
                        'duration_ns': op.duration_ns,
                    })
            params['C5'] = {
                'value': len(scaling_data),
                'unit': 'data points',
                'method': 'latency at various matrix sizes',
                'data': scaling_data[:100],  # sample
            }

        return params

    def _extract_mte_params(self) -> Dict[str, Any]:
        """提取 M1-M8 的实际值。"""
        params = {}

        # M1-M5: 各级 Buffer 带宽（来自 DMA/copy 操作）
        copy_ops = [op for op in self.mte_ops
                    if 'copy' in op.op_name.lower() or 'dma' in op.op_name.lower()
                    or 'memcpy' in op.op_name.lower()]

        for op in copy_ops:
            data_size = 1
            for dim in op.input_size:
                data_size *= dim
            data_size_bytes = data_size * 4  # FP32
            time_ns = op.duration_ns
            if time_ns > 0:
                bw_gbs = data_size_bytes / (time_ns * 1e-9) / 1e9

                # Classify by buffer level
                buffer_level = op.details.get('buffer', op.details.get('level', 'unknown'))
                param_id = None
                if 'l1' in str(buffer_level).lower():
                    if 'read' in op.op_name.lower():
                        param_id = 'M1'
                    elif 'write' in op.op_name.lower():
                        param_id = 'M2'
                elif 'l0a' in str(buffer_level).lower():
                    param_id = 'M3'
                elif 'l0b' in str(buffer_level).lower():
                    param_id = 'M4'
                elif 'l0c' in str(buffer_level).lower():
                    param_id = 'M5'

                if param_id and param_id not in params:
                    params[param_id] = {
                        'value': float(bw_gbs),
                        'unit': 'GB/s',
                        'method': f'data_size / duration for {buffer_level} operations',
                    }
                elif param_id:
                    # Take max bandwidth
                    params[param_id]['value'] = max(params[param_id]['value'], float(bw_gbs))

        # M6: HBM 访存延迟
        hbm_ops = [op for op in self.mte_ops
                   if 'hbm' in op.op_name.lower() or 'ddr' in op.op_name.lower()
                   or ('load' in op.op_name.lower() and op.details.get('level') == 'hbm')]
        if hbm_ops:
            latencies = [op.latency for op in hbm_ops]
            params['M6'] = {
                'value': float(np.median(latencies)),
                'unit': 'cycles',
                'method': 'median HBM load latency',
            }

        # M7: 各级 Buffer 容量
        # 从带宽悬崖推断：找到数据量阈值
        if copy_ops:
            params['M7'] = {
                'value': '见带宽悬崖分析',
                'unit': 'KB',
                'method': 'bandwidth cliff detection from copy ops',
                'note': 'L1 ~1024KB, L0A/L0B/L0C ~64KB each (Ascend 910B typical)',
            }

        # M8: 数据搬运启动开销
        # 对小传输外推到零
        small_copies = [op for op in copy_ops
                        if sum(op.input_size) < 4096] if copy_ops else []
        if small_copies:
            # 线性回归
            sizes = []
            times = []
            for op in small_copies:
                data_size = 1
                for dim in op.input_size:
                    data_size *= dim
                sizes.append(data_size * 4)  # bytes
                times.append(op.latency)  # cycles
            if len(sizes) >= 3:
                coeffs = np.polyfit(sizes, times, 1)
                startup_cycles = float(coeffs[1])
                params['M8'] = {
                    'value': startup_cycles,
                    'unit': 'cycles',
                    'method': 'linear extrapolation to zero size',
                    'slope': float(coeffs[0]),
                }

        return params

    def _compute_max_concurrent(self, ops: List[TraceOp]) -> int:
        """
        计算任意时刻的最大并发操作数。
        时间轴扫描：维护活跃操作集合，找最大大小。
        """
        if not ops:
            return 0

        events = []
        for op in ops:
            events.append((op.start_time, 'start', op))
            events.append((op.end_time, 'end', op))

        events.sort(key=lambda e: e[0])

        active = set()
        max_active = 0

        for time, event_type, op in events:
            if event_type == 'start':
                active.add(id(op))
                max_active = max(max_active, len(active))
            else:
                active.discard(id(op))

        return max_active


def generate_ground_truth(trace_path: str, output_path: str,
                          clock_freq_mhz: float = 1000.0):
    """从 trace 生成 ground truth 文件。"""
    print(f"[INFO] Loading trace from {trace_path}")

    if not os.path.exists(trace_path):
        print(f"[WARN] Trace file not found: {trace_path}")
        print("[INFO] Generating estimated ground truth based on Ascend 910B specs")

        # Provide estimated values based on known architecture specs
        estimated = {
            "_note": "这些是昇腾910B的估计参考值。实际值应由 trace 文件确定。",
            "_clock_freq_mhz": clock_freq_mhz,
            "S1": {"value": 4, "unit": "cycles",
                   "note": "标量ALU操作通常1-4周期，取典型值4"},
            "S2": {"value": 1.0, "unit": "ops/cycle",
                   "note": "标量单元全流水线化，每周期1条标量指令"},
            "S3": {"value": 30, "unit": "cycles",
                   "note": "标量L1命中约30周期，若无L1约200+周期"},
            "V1": {"value": 8, "unit": "cycles",
                   "note": "256-wide FP32向量加法，流水线深度~8"},
            "V2": {"value": 10, "unit": "cycles",
                   "note": "FP32乘法比加法多1-2周期"},
            "V3": {"value": 1.0, "unit": "ops/cycle",
                   "note": "向量单元每周期可发射1条向量指令"},
            "V4": {"value": 8, "unit": "条",
                   "note": "向量流水线深度，可同时容纳8条指令"},
            "V5": {"value": 2, "unit": "cycles",
                   "note": "向量寄存器文件读2周期，写1周期（bypass后）"},
            "C1": {"value": 128, "unit": "cycles",
                   "note": "16×16 FP16 tile矩阵乘，约128周期"},
            "C2": {"value": 32, "unit": "TFLOPS",
                   "note": "910B FP16峰值约32 TFLOPS"},
            "C3": {"value": 4, "unit": "条",
                   "note": "Cube Unit可同时处理4个tile的流水线"},
            "C4": {"value": 8, "unit": "cycles",
                   "note": "L0A/L0B写约8周期，L0C读约8周期"},
            "C5": {"value": "折线图", "unit": "—",
                   "note": "O(MNK)增长，tile边界处有阶梯"},
            "M1": {"value": 2000, "unit": "GB/s",
                   "note": "L1读取峰值带宽"},
            "M2": {"value": 2000, "unit": "GB/s",
                   "note": "L1写入峰值带宽"},
            "M3": {"value": 3000, "unit": "GB/s",
                   "note": "L0A带宽"},
            "M4": {"value": 3000, "unit": "GB/s",
                   "note": "L0B带宽"},
            "M5": {"value": 3000, "unit": "GB/s",
                   "note": "L0C带宽"},
            "M6": {"value": 300, "unit": "cycles",
                   "note": "HBM首字延迟约300周期"},
            "M7": {"value": "L1:1024KB, L0A/B/C:64KB", "unit": "—",
                   "note": "各buffer可用容量"},
            "M8": {"value": 50, "unit": "cycles",
                   "note": "DMA启动固定开销约50周期"},
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(estimated, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Estimated ground truth saved to {output_path}")
        return estimated

    # Normal trace processing
    raw_ops = load_trace(trace_path)
    ops = parse_ops(raw_ops)
    print(f"[INFO] Loaded {len(ops)} operations from trace")

    extractor = GroundTruthExtractor(ops, clock_freq_mhz)
    ground_truth = extractor.extract_all()

    # Add metadata
    ground_truth['_metadata'] = {
        'total_ops': len(ops),
        'clock_freq_mhz': clock_freq_mhz,
        'categories': {
            'scalar': len(extractor.scalar_ops),
            'vector': len(extractor.vector_ops),
            'cube': len(extractor.cube_ops),
            'mte': len(extractor.mte_ops),
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Ground truth saved to {output_path}")

    return ground_truth


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="从 simulator trace 提取参数实际值"
    )
    parser.add_argument('--trace', default='simulator_trace/trace.json',
                        help='Path to trace.json')
    parser.add_argument('--output', default='results/ground_truth.json',
                        help='Output path for ground truth')
    parser.add_argument('--freq', type=float, default=1000.0,
                        help='Clock frequency in MHz')

    args = parser.parse_args()

    gt = generate_ground_truth(args.trace, args.output, args.freq)

    print("\n" + "=" * 60)
    print("  GROUND TRUTH SUMMARY")
    print("=" * 60)
    for key, val in sorted(gt.items()):
        if key.startswith('_'):
            continue
        if isinstance(val, dict) and 'value' in val:
            print(f"  {key}: {val['value']} {val.get('unit', '')}")
        else:
            print(f"  {key}: {val}")


if __name__ == '__main__':
    main()
