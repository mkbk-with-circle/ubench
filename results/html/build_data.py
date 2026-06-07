#!/usr/bin/env python3
"""
build_data.py — 从 summary.csv 和逐 benchmark CSV 生成 EMBEDDED_DATA JSON，
可直接输出到 stdout，也可 --embed 写回 index.html。

用法:
  python3 results/html/build_data.py                  # 打印 JSON
  python3 results/html/build_data.py --embed           # 写回 index.html
  python3 results/html/build_data.py --embed --html path/to/index.html
"""
import argparse, csv, json, os, re, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent          # ubench/
RESULTS_DIR = SCRIPT_DIR.parent              # ubench/results/

# ── kernel 常量（从 kernel 源码抄录，不运行时解析） ─────────────────────────
KERNEL_CONSTANTS = {
    "mte_copy_bw": {
        "bytes_per_repeat_per_block": 2048,   # kTileBytes=1024, GM→UB + UB→GM
        "blocks": 4,
        "label": "GM↔UB 带宽",
        "tooltip": "每 repeat: 4 blocks × 1024B×2 (GM→UB→GM) = 8192 B",
        "lab_mapping": "M1≈ GM↔UB 带宽探针，非纯 L2→L1",
    },
    "mte_startup_latency": {
        "copy_bytes": 32,
        "label": "32B 拷贝延迟",
        "tooltip": "每 repeat: 完整 32B DataCopy GM→UB→GM 往返",
        "lab_mapping": "M8≈ DataCopy 启动+往返，非纯 DMA 启动开销",
    },
    "mte_granularity": {
        "copy_bytes": 32,  # mode=0 → 32B
        "label": "DataCopy 粒度",
        "tooltip": "mode=0 时 32B 最小拷贝粒度",
        "lab_mapping": "非 Lab2 M7 容量曲线；仅测最小粒度延迟",
    },
    "vector_throughput": {
        "lanes": 4,
        "elems": 256,
        "blocks": 4,
        "flops_per_elem": 1,  # 1 FP32 Add per element
        "label": "FP32 向量吞吐",
        "tooltip": "每 repeat: 4 blocks × 4 lanes × 256 FP32 Add = 4096 ops",
        "lab_mapping": "V3≈ 向量 Add 吞吐（带 DataCopy/PipeBarrier，未满占）",
    },
    "vector_add_latency": {
        "label": "FP32 Add RAW 延迟",
        "tooltip": "依赖链: Add + PipeBarrier<PIPE_V>，测 RAW 冒险停顿",
        "lab_mapping": "V1≈ 向量算术延迟",
    },
    "vector_mul_latency": {
        "label": "FP32 Mul RAW 延迟",
        "tooltip": "依赖链: Mul + PipeBarrier<PIPE_V>，测 RAW 冒险停顿",
        "lab_mapping": "V2≈ 向量算术延迟",
    },
    "vector_pipeline_depth": {
        "label": "向量管线深度",
        "tooltip": "依赖 Add + 独立 gap ops，测管线隐藏能力",
        "lab_mapping": "V4≈ 管线深度探针",
    },
    "scalar_arith_latency": {
        "label": "标量算术延迟",
        "tooltip": "LCG: x = x*1664525 + 1013904223，R²≈0.93 低可信度",
        "lab_mapping": "S1≈ 标量延迟（实验性，不作为正式 KPI）",
        "unreliable": True,
        "unreliable_reason": "R²=0.932 < 0.95；baseline 与 target 成本接近，adjusted 有负值",
    },
    "scalar_branch_overhead": {
        "label": "分支开销",
        "tooltip": "条件分支 vs 无分支 baseline",
        "lab_mapping": "S2≈ 分支探针（非 S3 访存延迟）",
    },
}

CUBE_BENCHMARKS = ["cube_tile_latency", "cube_throughput", "cube_scaling"]


def read_summary(path: Path) -> dict:
    """读 summary.csv，返回 {benchmark_name: row_dict}"""
    rows = {}
    if not path.exists():
        return rows
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[row["benchmark"]] = row
    return rows


def read_detail_csv(path: Path) -> list:
    """读单个 benchmark CSV 的 repeat sweep 行"""
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def derive_metric(name: str, slope_us: float, meta: dict) -> dict:
    """根据 benchmark 名和 kernel 常量推导物理指标"""
    kc = KERNEL_CONSTANTS.get(name, {})
    result = {}

    if name == "mte_copy_bw":
        bprpb = kc.get("bytes_per_repeat_per_block", 2048)
        blocks = kc.get("blocks", 4)
        total_bytes = bprpb * blocks  # 8192
        bw_gbs = total_bytes / (slope_us * 1e-6) / 1e9
        result["bandwidth_gbs"] = round(bw_gbs, 1)
        result["primary_value"] = f"{bw_gbs:.1f}"
        result["primary_unit"] = "GB/s"
        result["primary_label"] = "聚合带宽"

    elif name == "mte_startup_latency":
        ns = slope_us * 1000
        result["latency_ns"] = round(ns, 1)
        result["primary_value"] = f"{ns:.1f}"
        result["primary_unit"] = "ns"
        result["primary_label"] = "32B 拷贝延迟"

    elif name == "mte_granularity":
        ns = slope_us * 1000
        result["latency_ns"] = round(ns, 1)
        result["primary_value"] = f"{ns:.1f}"
        result["primary_unit"] = "ns"
        result["primary_label"] = "最小粒度延迟"

    elif name == "vector_throughput":
        lanes = kc.get("lanes", 4)
        elems = kc.get("elems", 256)
        blocks = kc.get("blocks", 4)
        flops_per = kc.get("flops_per_elem", 1)
        total_ops = blocks * lanes * elems * flops_per  # 4096
        gflops = total_ops / (slope_us * 1e-6) / 1e9
        ops_per_cycle = (blocks * lanes) / (slope_us * 1000)  # @1GHz
        result["gflops"] = round(gflops, 1)
        result["ops_per_cycle"] = round(ops_per_cycle, 3)
        result["total_ops_per_repeat"] = total_ops
        result["primary_value"] = f"{gflops:.1f}"
        result["primary_unit"] = "GFLOPS"
        result["primary_label"] = "FP32 吞吐"

    elif name in ("vector_add_latency", "vector_mul_latency", "vector_pipeline_depth"):
        ns = slope_us * 1000
        result["latency_ns"] = round(ns, 1)
        result["primary_value"] = f"{ns:.1f}"
        result["primary_unit"] = "ns"
        result["primary_label"] = kc.get("label", "延迟")

    elif name == "scalar_arith_latency":
        ns = slope_us * 1000
        result["latency_ns"] = round(ns, 2)
        result["primary_value"] = f"{ns:.2f}"
        result["primary_unit"] = "ns"
        result["primary_label"] = "LCG 步进延迟"

    elif name == "scalar_branch_overhead":
        ns = slope_us * 1000
        result["latency_ns"] = round(ns, 1)
        result["primary_value"] = f"{ns:.1f}"
        result["primary_unit"] = "ns"
        result["primary_label"] = "分支开销"

    else:
        result["primary_value"] = f"{slope_us:.4f}"
        result["primary_unit"] = "μs/op"
        result["primary_label"] = "斜率"

    return result


def build_embedded_data() -> dict:
    """构建完整的 EMBEDDED_DATA JSON"""
    summary = read_summary(RESULTS_DIR / "summary.csv")

    benchmarks = []
    for name in sorted(KERNEL_CONSTANTS.keys()):
        row = summary.get(name)
        kc = KERNEL_CONSTANTS[name]

        if row is None:
            # cube 或缺失
            benchmarks.append({
                "name": name,
                "status": "skipped" if name in CUBE_BENCHMARKS else "missing",
                "label": kc.get("label", name),
                "tooltip": kc.get("tooltip", ""),
                "lab_mapping": kc.get("lab_mapping", ""),
            })
            continue

        slope = float(row.get("adjusted_slope_us", 0))
        r2 = float(row.get("adjusted_r2", 0))
        raw_slope = float(row.get("raw_slope_us", 0))
        raw_r2 = float(row.get("raw_r2", 0))

        derived = derive_metric(name, slope, kc)

        detail_rows = read_detail_csv(RESULTS_DIR / f"{name}.csv")
        sweep = []
        for dr in detail_rows:
            sweep.append({
                "repeat": int(dr.get("repeat", 0)),
                "total_us": float(dr.get("total_us", 0)),
                "baseline_us": float(dr.get("baseline_us", 0)),
                "adjusted_us": float(dr.get("adjusted_us", 0)),
            })

        benchmarks.append({
            "name": name,
            "status": "ok",
            "label": kc.get("label", name),
            "tooltip": kc.get("tooltip", ""),
            "lab_mapping": kc.get("lab_mapping", ""),
            "unreliable": kc.get("unreliable", False),
            "unreliable_reason": kc.get("unreliable_reason", ""),
            "slope_us": slope,
            "intercept_us": float(row.get("adjusted_intercept_us", 0)),
            "r2": r2,
            "raw_slope_us": raw_slope,
            "raw_r2": raw_r2,
            "derived": derived,
            "sweep": sweep,
        })

    # cube 占位
    for cname in CUBE_BENCHMARKS:
        if not any(b["name"] == cname for b in benchmarks):
            benchmarks.append({
                "name": cname,
                "status": "skipped",
                "label": cname.replace("_", " "),
                "tooltip": "310P Matmul 模板挂起，已跳过",
                "lab_mapping": "Cube N/A — 待 910B 验证",
            })

    return {
        "meta": {
            "platform": "Ascend 310P3",
            "cann_version": "8.2.RC1",
            "date": "2026-06-07",
            "run_params": {
                "device": 0,
                "warmup": 3,
                "iters": 10,
                "blocks": 4,
                "repeats_sweep": [500, 1000, 2500, 5000],
                "size_bytes": 65536,
            },
            "kernel_constants": KERNEL_CONSTANTS,
            "disclaimer": (
                "本结果基于 Ascend 310P3 (m200) 实测。"
                "Cube 基准因 Matmul 模板兼容性问题挂起，已跳过。"
                "本结果不能与 Lab2 910B 助教标定值直接对比；"
                "Lab 参数映射为近似（M1 实为 GM↔UB 带宽探针等）。"
                "scalar_arith_latency R²<0.95 标记为不可靠/实验性。"
            ),
        },
        "benchmarks": benchmarks,
    }


def embed_into_html(html_path: Path, data_json: str):
    """将 EMBEDDED_DATA JSON 写回 index.html 的 <script id="EMBEDDED_DATA"> 块"""
    html = html_path.read_text(encoding="utf-8")

    block = f'<script id="EMBEDDED_DATA" type="application/json">\n{data_json}\n</script>'

    if 'id="EMBEDDED_DATA"' in html:
        html = re.sub(
            r'<script id="EMBEDDED_DATA"[^>]*>.*?</script>',
            block, html, flags=re.DOTALL
        )
    else:
        # 插入到 </head> 前
        html = html.replace('</head>', f'{block}\n</head>')

    html_path.write_text(html, encoding="utf-8")
    print(f"[OK] EMBEDDED_DATA written to {html_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Build EMBEDDED_DATA for ubench HTML report")
    parser.add_argument("--embed", action="store_true", help="Write JSON back into index.html")
    parser.add_argument("--html", type=str, default=str(SCRIPT_DIR / "index.html"),
                        help="Path to index.html (default: results/html/index.html)")
    args = parser.parse_args()

    data = build_embedded_data()
    data_json = json.dumps(data, indent=2, ensure_ascii=False)

    if args.embed:
        embed_into_html(Path(args.html), data_json)
    else:
        print(data_json)


if __name__ == "__main__":
    main()
