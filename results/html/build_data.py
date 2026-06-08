#!/usr/bin/env python3
"""从 results/*.csv 生成 EMBEDDED_DATA，并写回 index.html。

用法: python3 results/html/build_data.py --embed
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent

# 910B AI Core 主频（用于 ns/us → cycles 换算）
FREQ_GHZ = 1.5

KERNEL_CONSTANTS = {
    "mte_copy_bw": {
        "bytes_per_repeat_per_block": 2048,
        "blocks": 4,
        "label": "GM↔UB 拷贝带宽",
        "tooltip": "每 repeat: 4×1024B×2 (GM→UB→GM) = 8192B",
    },
    "mte_startup_latency": {
        "copy_bytes": 32,
        "label": "32B DataCopy 往返",
        "tooltip": "完整 32B GM→UB→GM，含 barrier",
    },
    "mte_granularity": {
        "copy_bytes": 32,
        "label": "最小搬运粒度",
        "tooltip": "mode=0 → 32B DataCopy",
    },
    "vector_throughput": {
        "lanes": 4, "elems": 256, "blocks": 4, "flops_per_elem": 1,
        "label": "FP32 向量吞吐",
        "tooltip": "4 blocks×4 lanes×256 FP32 Add / repeat",
    },
    "vector_add_latency": {"label": "FP32 Add RAW", "tooltip": "Add + PipeBarrier"},
    "vector_mul_latency": {"label": "FP32 Mul RAW", "tooltip": "Mul + PipeBarrier"},
    "vector_pipeline_depth": {"label": "流水线深度探针", "tooltip": "依赖 Add + gap ops"},
    "scalar_arith_latency": {
        "label": "标量算术链",
        "tooltip": "16 次链式乘加 RAW 依赖",
        "ops_per_repeat": 16,
    },
    "scalar_branch_overhead": {"label": "分支开销探针", "tooltip": "条件分支 vs 无分支"},
}

# Lab2 必测 21 项（见 lab2.pdf §4.2）
LAB2_PARAM_DEFS = [
    # Scalar
    {"id": "S1", "cat": "Scalar", "name": "标量算术指令延迟", "unit": "cycles",
     "bench": "scalar_arith_latency", "kind": "scalar_chain_cycles"},
    {"id": "S2", "cat": "Scalar", "name": "标量单元吞吐率", "unit": "ops/cycle",
     "bench": None, "kind": "missing"},
    {"id": "S3", "cat": "Scalar", "name": "标量访存延迟", "unit": "cycles",
     "bench": None, "kind": "missing"},
    # Vector
    {"id": "V1", "cat": "Vector", "name": "FP32 向量加法延迟", "unit": "cycles",
     "bench": "vector_add_latency", "kind": "latency_cycles"},
    {"id": "V2", "cat": "Vector", "name": "FP32 向量乘法延迟", "unit": "cycles",
     "bench": "vector_mul_latency", "kind": "latency_cycles"},
    {"id": "V3", "cat": "Vector", "name": "向量单元吞吐率", "unit": "ops/cycle",
     "bench": "vector_throughput", "kind": "vec_throughput"},
    {"id": "V4", "cat": "Vector", "name": "向量流水线深度", "unit": "条",
     "bench": "vector_pipeline_depth", "kind": "pipeline_probe"},
    {"id": "V5", "cat": "Vector", "name": "向量寄存器访问延迟", "unit": "cycles",
     "bench": None, "kind": "missing"},
    # Cube
    {"id": "C1", "cat": "Cube", "name": "矩阵乘延迟（单 tile）", "unit": "cycles",
     "bench": "cube_tile_latency", "kind": "cube_skipped"},
    {"id": "C2", "cat": "Cube", "name": "矩阵乘吞吐率", "unit": "TFLOPS",
     "bench": "cube_throughput", "kind": "cube_skipped"},
    {"id": "C3", "cat": "Cube", "name": "矩阵乘流水线深度", "unit": "条",
     "bench": None, "kind": "missing"},
    {"id": "C4", "cat": "Cube", "name": "L0A/L0B/L0C 访问延迟", "unit": "cycles",
     "bench": None, "kind": "missing"},
    {"id": "C5", "cat": "Cube", "name": "矩阵规模延迟缩放", "unit": "—",
     "bench": "cube_scaling", "kind": "cube_skipped"},
    # MTE
    {"id": "M1", "cat": "MTE", "name": "L1 Buffer 读取带宽", "unit": "GB/s",
     "bench": "mte_copy_bw", "kind": "mte_bw", "note": "探针为 GM↔UB 往返，近似 M1"},
    {"id": "M2", "cat": "MTE", "name": "L1 Buffer 写入带宽", "unit": "GB/s",
     "bench": None, "kind": "missing"},
    {"id": "M3", "cat": "MTE", "name": "L0A 带宽", "unit": "GB/s",
     "bench": None, "kind": "missing"},
    {"id": "M4", "cat": "MTE", "name": "L0B 带宽", "unit": "GB/s",
     "bench": None, "kind": "missing"},
    {"id": "M5", "cat": "MTE", "name": "L0C 带宽", "unit": "GB/s",
     "bench": None, "kind": "missing"},
    {"id": "M6", "cat": "MTE", "name": "DDR/HBM 访存延迟", "unit": "cycles",
     "bench": None, "kind": "missing"},
    {"id": "M7", "cat": "MTE", "name": "各级 Buffer 容量", "unit": "KB/MB",
     "bench": "mte_granularity", "kind": "granularity_partial",
     "note": "仅测 32B 最小粒度，非容量曲线"},
    {"id": "M8", "cat": "MTE", "name": "数据搬运启动开销", "unit": "cycles",
     "bench": "mte_startup_latency", "kind": "startup_cycles",
     "note": "含 32B 完整往返，非纯 DMA 启动"},
]

CUBE_BENCHES = {"cube_tile_latency", "cube_throughput", "cube_scaling"}


def read_summary(path: Path) -> dict:
    rows = {}
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["benchmark"]] = row
    return rows


def read_detail_csv(path: Path) -> list:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def us_to_cycles(slope_us: float) -> float:
    return round(slope_us * FREQ_GHZ * 1000, 2)


def derive_benchmark(name: str, slope_us: float) -> dict:
    kc = KERNEL_CONSTANTS.get(name, {})
    d = {"slope_us": slope_us}

    if name == "mte_copy_bw":
        total_b = kc["bytes_per_repeat_per_block"] * kc["blocks"]
        bw = total_b / (slope_us * 1e-6) / 1e9
        d.update(bandwidth_gbs=round(bw, 1), primary_value=f"{bw:.1f}", primary_unit="GB/s")
    elif name == "vector_throughput":
        ops = kc["blocks"] * kc["lanes"] * kc["elems"]
        gflops = ops / (slope_us * 1e-6) / 1e9
        vec_inst = kc["blocks"] * kc["lanes"]
        opc = vec_inst / (slope_us * FREQ_GHZ * 1000)
        d.update(gflops=round(gflops, 1), ops_per_cycle=round(opc, 3),
                 total_ops=ops, primary_value=f"{opc:.3f}", primary_unit="ops/cycle")
    elif name in ("vector_add_latency", "vector_mul_latency", "vector_pipeline_depth",
                  "mte_startup_latency", "mte_granularity", "scalar_branch_overhead"):
        ns = slope_us * 1000
        cyc = us_to_cycles(slope_us)
        d.update(latency_ns=round(ns, 2), cycles=cyc,
                 primary_value=f"{cyc:.1f}", primary_unit="cycles")
    elif name == "scalar_arith_latency":
        ops = kc.get("ops_per_repeat", 16)
        cyc_per_repeat = us_to_cycles(slope_us)
        cyc_per_op = round(cyc_per_repeat / ops, 2)
        d.update(cycles_per_op=cyc_per_op, cycles_per_repeat=cyc_per_repeat,
                 primary_value=f"{cyc_per_op:.2f}", primary_unit="cycles/op")
    return d


def resolve_lab2_param(defn: dict, bench_by_name: dict) -> dict:
    p = {**defn, "status": "missing", "value": None, "display": "—", "r2": None, "bench_status": None}

    if defn["kind"] == "missing":
        p["status"] = "missing"
        return p

    if defn["kind"] == "cube_skipped":
        p["status"] = "skipped"
        p["display"] = "N/A"
        p["note"] = p.get("note") or "AIC 二进制注册失败 (CANN 9.0)"
        return p

    bench_name = defn.get("bench")
    b = bench_by_name.get(bench_name)
    if not b or b.get("status") != "ok":
        p["status"] = "missing"
        return p

    derived = b["derived"]
    slope = b["slope_us"]
    r2 = b["r2"]
    p["r2"] = r2
    p["bench_status"] = "ok"

    kind = defn["kind"]
    if kind == "mte_bw":
        p["status"] = "partial"
        p["value"] = derived["bandwidth_gbs"]
        p["display"] = f"{derived['bandwidth_gbs']} GB/s"
    elif kind == "startup_cycles":
        p["status"] = "partial"
        p["value"] = derived["cycles"]
        p["display"] = f"{derived['cycles']} cycles ({derived['latency_ns']:.0f} ns)"
    elif kind == "granularity_partial":
        p["status"] = "partial"
        p["value"] = derived["latency_ns"]
        p["display"] = f"32B 延迟 {derived['latency_ns']:.0f} ns（非容量曲线）"
    elif kind == "latency_cycles":
        p["status"] = "measured"
        p["value"] = derived["cycles"]
        p["display"] = f"{derived['cycles']} cycles ({derived['latency_ns']:.1f} ns)"
    elif kind == "vec_throughput":
        p["status"] = "measured"
        p["value"] = derived["ops_per_cycle"]
        p["display"] = f"{derived['ops_per_cycle']} vec-inst/cycle ({derived['gflops']} GFLOPS)"
    elif kind == "pipeline_probe":
        p["status"] = "partial"
        p["value"] = derived["cycles"]
        p["display"] = f"探针 {derived['cycles']} cycles（≈V1，gap 被隐藏）"
    elif kind == "scalar_chain_cycles":
        unreliable = r2 < 0.95
        p["status"] = "partial" if unreliable else "measured"
        p["value"] = derived["cycles_per_op"]
        p["display"] = f"{derived['cycles_per_op']} cycles/op (16 链/repeat)"
    else:
        p["status"] = "measured"
        p["display"] = derived.get("primary_value", "—")

    return p


def build_embedded_data() -> dict:
    summary = read_summary(RESULTS_DIR / "summary.csv")

    # 从 CSV 推断 sweep 与运行参数
    repeats_sweep = []
    sample = read_detail_csv(RESULTS_DIR / "mte_copy_bw.csv")
    if sample:
        repeats_sweep = sorted({int(r["repeat"]) for r in sample})

    run_params = {
        "device": 0, "warmup": 5, "iters": 20, "blocks": 4,
        "repeats_sweep": repeats_sweep or [1000, 2000, 5000, 10000],
        "size_bytes": 65536,
    }
    if summary:
        row = next(iter(summary.values()))
        run_params["device"] = int(row.get("device", 0))
        run_params["blocks"] = int(row.get("blocks", 4))
        run_params["size_bytes"] = int(row.get("size_bytes", 65536))

    benchmarks = []
    for name in sorted(KERNEL_CONSTANTS.keys()):
        kc = KERNEL_CONSTANTS[name]
        row = summary.get(name)
        if not row:
            benchmarks.append({"name": name, "status": "missing", "label": kc.get("label", name)})
            continue

        slope = float(row["adjusted_slope_us"])
        r2 = float(row["adjusted_r2"])
        derived = derive_benchmark(name, slope)
        sweep = []
        for dr in read_detail_csv(RESULTS_DIR / f"{name}.csv"):
            sweep.append({
                "repeat": int(dr["repeat"]),
                "total_us": float(dr["total_us"]),
                "baseline_us": float(dr["baseline_us"]),
                "adjusted_us": float(dr["adjusted_us"]),
            })

        benchmarks.append({
            "name": name, "status": "ok", "label": kc.get("label", name),
            "tooltip": kc.get("tooltip", ""), "unreliable": r2 < 0.95,
            "slope_us": slope, "intercept_us": float(row["adjusted_intercept_us"]),
            "r2": r2, "raw_slope_us": float(row["raw_slope_us"]), "raw_r2": float(row["raw_r2"]),
            "derived": derived, "sweep": sweep,
        })

    bench_by_name = {b["name"]: b for b in benchmarks}

    lab2_params = [resolve_lab2_param(d, bench_by_name) for d in LAB2_PARAM_DEFS]

    for cname in CUBE_BENCHES:
        if cname not in bench_by_name:
            benchmarks.append({
                "name": cname, "status": "skipped", "label": cname.replace("_", " "),
            })

    measured = sum(1 for p in lab2_params if p["status"] == "measured")
    partial = sum(1 for p in lab2_params if p["status"] == "partial")
    skipped = sum(1 for p in lab2_params if p["status"] == "skipped")
    missing = sum(1 for p in lab2_params if p["status"] == "missing")

    return {
        "meta": {
            "platform": "Ascend 910B",
            "chip": "Ascend910_9362",
            "cann_version": "9.0.0",
            "date": "2026-06-08",
            "freq_ghz": FREQ_GHZ,
            "run_params": run_params,
            "kernel_constants": KERNEL_CONSTANTS,
            "coverage": {
                "lab2_total": 21,
                "measured": measured,
                "partial": partial,
                "skipped": skipped,
                "missing": missing,
                "benchmarks_ok": sum(1 for b in benchmarks if b.get("status") == "ok"),
                "benchmarks_total": 12,
            },
            "disclaimer": (
                "Ascend 910B (CANN 9.0) 实测。9/12 μbench 通过；Cube 三项因 AIC 注册失败跳过。"
                "部分 Lab2 参数为近似映射（M1/M8/M7 等）。未测项需在报告中说明计划。"
            ),
        },
        "lab2_params": lab2_params,
        "benchmarks": benchmarks,
    }


def embed_into_html(html_path: Path, data_json: str):
    html = html_path.read_text(encoding="utf-8")
    block = f'<script id="EMBEDDED_DATA" type="application/json">\n{data_json}\n</script>'
    html = re.sub(
        r'<script id="EMBEDDED_DATA"[^>]*>.*?</script>',
        block, html, count=1, flags=re.DOTALL,
    )
    html_path.write_text(html, encoding="utf-8")
    print(f"[OK] embedded → {html_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--html", default=str(SCRIPT_DIR / "index.html"))
    args = parser.parse_args()

    data = build_embedded_data()
    js = json.dumps(data, indent=2, ensure_ascii=False)
    if args.embed:
        embed_into_html(Path(args.html), js)
    else:
        print(js)


if __name__ == "__main__":
    main()
