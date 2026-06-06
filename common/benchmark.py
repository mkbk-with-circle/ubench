#!/usr/bin/env python3
"""
Ascend NPU Microbenchmark Harness
Provides common infrastructure for precise timing measurements on Ascend NPU.

Key design principles:
1. Isolation: Each benchmark tests ONE specific hardware operation
2. Controllability: Precise control over data size, iterations, and launch configuration
3. Repeatability: Multiple warm-up runs + measurement iterations with statistics
4. Verifiability: Results validated against theoretical peaks where possible
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import statistics
import json
import os

try:
    import torch
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False
    print("[WARN] torch_npu not available - running in simulation mode")


@dataclass
class BenchResult:
    """Result of a single benchmark measurement."""
    param_id: str
    param_name: str
    category: str
    value: float
    unit: str
    raw_times: List[float] = field(default_factory=list)
    num_iterations: int = 0
    num_warmup: int = 0
    std_dev: float = 0.0
    cv: float = 0.0  # coefficient of variation
    median: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    notes: str = ""

    def to_dict(self):
        return {
            "param_id": self.param_id,
            "param_name": self.param_name,
            "category": self.category,
            "value": self.value,
            "unit": self.unit,
            "raw_times": self.raw_times,
            "num_iterations": self.num_iterations,
            "num_warmup": self.num_warmup,
            "std_dev": self.std_dev,
            "cv": self.cv,
            "median": self.median,
            "p25": self.p25,
            "p75": self.p75,
            "notes": self.notes,
        }


class AscendBenchmark:
    """
    Base class for all Ascend NPU microbenchmarks.

    Usage pattern:
    1. Subclass and implement bench_kernel() to define the measurement operation
    2. Call run() with appropriate parameters
    3. Results are automatically collected with statistics
    """

    def __init__(self, param_id: str, param_name: str, category: str, device_id: int = 0):
        self.param_id = param_id
        self.param_name = param_name
        self.category = category
        self.device_id = device_id
        self.has_npu = HAS_NPU

        if HAS_NPU:
            try:
                torch.npu.set_device(device_id)
            except Exception:
                pass

    def synchronize(self):
        """Synchronize NPU device."""
        if HAS_NPU:
            torch.npu.synchronize()

    def bench_kernel(self, *args, **kwargs) -> float:
        """
        Override this method to implement the core measurement operation.
        Should return the metric of interest (e.g., time in ms, bandwidth in GB/s).

        This is the operation being measured - it should be as isolated as possible.
        """
        raise NotImplementedError("Subclass must implement bench_kernel()")

    def run(
        self,
        num_warmup: int = 5,
        num_iterations: int = 100,
        warmup_args: tuple = (),
        warmup_kwargs: Optional[dict] = None,
        bench_args: tuple = (),
        bench_kwargs: Optional[dict] = None,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
    ) -> BenchResult:
        """
        Execute the benchmark with warmup and measurement phases.

        Args:
            num_warmup: Number of warmup iterations (excluded from results)
            num_iterations: Number of measurement iterations
            warmup_args/kwargs: Arguments for warmup kernel calls
            bench_args/kwargs: Arguments for measurement kernel calls
            preprocess_fn: Called once before all iterations for setup
            postprocess_fn: Called once after all iterations for cleanup
        """
        if bench_kwargs is None:
            bench_kwargs = {}
        if warmup_kwargs is None:
            warmup_kwargs = {}

        # Phase 0: Preprocessing
        if preprocess_fn:
            preprocess_fn()

        # Phase 1: Warmup
        for _ in range(num_warmup):
            self.bench_kernel(*warmup_args, **warmup_kwargs)
        self.synchronize()

        # Phase 2: Measurement
        raw_times = []
        for _ in range(num_iterations):
            elapsed = self.bench_kernel(*bench_args, **bench_kwargs)
            raw_times.append(elapsed)
        self.synchronize()

        # Phase 3: Postprocessing
        if postprocess_fn:
            postprocess_fn()

        # Phase 4: Statistical analysis
        return self._compute_statistics(
            raw_times, num_warmup, num_iterations
        )

    def _compute_statistics(
        self, raw_times: List[float], num_warmup: int, num_iterations: int
    ) -> BenchResult:
        """Compute statistical measures from raw timing data."""
        arr = np.array(raw_times)

        # Remove outliers using IQR method
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered = arr[(arr >= lower_bound) & (arr <= upper_bound)]

        if len(filtered) == 0:
            filtered = arr  # fallback if all filtered out

        median_val = float(np.median(arr))
        mean_val = float(np.mean(filtered))
        std_val = float(np.std(filtered))
        cv_val = std_val / mean_val if mean_val > 0 else 0.0

        return BenchResult(
            param_id=self.param_id,
            param_name=self.param_name,
            category=self.category,
            value=mean_val,
            unit="ms",  # subclasses override
            raw_times=raw_times,
            num_iterations=num_iterations,
            num_warmup=num_warmup,
            std_dev=std_val,
            cv=cv_val,
            median=median_val,
            p25=float(np.percentile(arr, 25)),
            p75=float(np.percentile(arr, 75)),
        )


class EventTimer:
    """
    High-precision NPU event-based timer.
    Uses Ascend device events for accurate GPU-side timing,
    avoiding CPU-side launch overhead.
    """

    def __init__(self):
        self.start_event = None
        self.end_event = None
        if HAS_NPU:
            self.start_event = torch.npu.Event(enable_timing=True)
            self.end_event = torch.npu.Event(enable_timing=True)
        else:
            self._cpu_start = 0.0
            self._cpu_end = 0.0

    def record_start(self, stream=None):
        """Record start event."""
        if HAS_NPU:
            if stream is not None:
                self.start_event.record(stream)
            else:
                self.start_event.record()
        else:
            self._cpu_start = time.perf_counter()

    def record_end(self, stream=None):
        """Record end event."""
        if HAS_NPU:
            if stream is not None:
                self.end_event.record(stream)
            else:
                self.end_event.record()
        else:
            self._cpu_end = time.perf_counter()

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if HAS_NPU:
            torch.npu.synchronize()
            return self.start_event.elapsed_time(self.end_event)
        else:
            return (self._cpu_end - self._cpu_start) * 1000.0


def save_results(results: List[BenchResult], filepath: str):
    """Save all benchmark results to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = {
        "results": [r.to_dict() for r in results],
        "summary": {
            "total_params": len(results),
            "categories": list(set(r.category for r in results)),
        }
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Results saved to {filepath}")


def print_result(result: BenchResult):
    """Pretty-print a benchmark result."""
    print(f"\n{'='*60}")
    print(f"  [{result.param_id}] {result.param_name}")
    print(f"  Category: {result.category}")
    print(f"  Value:    {result.value:.4f} {result.unit}")
    print(f"  Median:   {result.median:.4f} {result.unit}")
    print(f"  Std Dev:  {result.std_dev:.4f}")
    print(f"  CV:       {result.cv:.2%}")
    print(f"  P25-P75:  [{result.p25:.4f}, {result.p75:.4f}]")
    print(f"  Iters:    {result.num_iterations} (warmup={result.num_warmup})")
    if result.notes:
        print(f"  Notes:    {result.notes}")
    print(f"{'='*60}")
