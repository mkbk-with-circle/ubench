#!/usr/bin/env bash
# ============================================================
#  一键运行全部测试并生成报告
#  用法: bash scripts/run_and_report.sh [--quick]
#
#  等价于:
#    bash scripts/run_all.sh
#    bash scripts/generate_report.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  昇腾 NPU μbench 一键运行 + 报告生成"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# 激活环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null

# Step 1: 运行测试
echo ">>> Step 1: 运行基准测试"
echo ""
bash "$SCRIPT_DIR/run_all.sh" "$@"

echo ""
echo ">>> Step 2: 生成报告"
echo ""
bash "$SCRIPT_DIR/generate_report.sh"

echo ""
echo "============================================================"
echo "  ✅ 全部完成！"
echo ""
echo "  产出文件:"
echo "    results/all_results.json       — 汇总结果"
echo "    results/extracted_params.json  — 关键参数"
echo "    results/c5_scaling_chart.png   — C5 折线图"
echo "    results/dashboard.html         — HTML 仪表板"
echo "    report_template.md             — 实验报告"
echo ""
echo "  查看仪表板:"
echo "    open results/dashboard.html"
echo "============================================================"
