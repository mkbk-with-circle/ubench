#!/usr/bin/env bash
# ============================================================
#  生成可视化报告脚本
#  用法: bash scripts/generate_report.sh
#
#  功能:
#    1. 从 results/*.json 提取关键参数
#    2. 生成 C5 折线图 (PNG)
#    3. 生成 HTML 仪表板
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "  生成可视化报告"
echo "============================================================"

# 激活环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
echo "✅ CANN 环境已激活"

cd "$PROJECT_DIR"

# 检查结果文件
if [ ! -f results/all_results.json ]; then
    echo "❌ 未找到 results/all_results.json，请先运行 bash scripts/run_all.sh"
    exit 1
fi

echo ""
echo "[1/3] 提取关键参数..."
python3 -c "
import json, re, numpy as np

with open('results/all_results.json') as f:
    data = json.load(f)
results = data['results']
params = {}

# S1-S3
for pid in ['S1', 'S2', 'S3']:
    items = [r for r in results if r['param_id'] == pid]
    if items:
        if pid == 'S2':
            medians = [r['median'] for r in items if r['median'] > 0]
            params[pid] = max(medians) if medians else 0
        else:
            params[pid] = min(r['value'] for r in items)

# V1-V5
for pid in ['V1', 'V2']:
    items = [r for r in results if r['param_id'] == pid]
    if items: params[pid] = min(r['value'] for r in items)

v3 = [r for r in results if r['param_id'] == 'V3']
if v3: params['V3'] = max(r['median'] for r in v3 if r['median'] > 0)

for pid in ['V5_READ', 'V5_WRITE']:
    items = [r for r in results if r['param_id'] == pid]
    if items: params[pid] = min(r['median'] for r in items)

# C1-C4
c1 = [r for r in results if r['param_id'] == 'C1' and 'tile=16' in (r.get('notes','') or '')]
if c1: params['C1'] = c1[0]['median']

c2 = [r for r in results if r['param_id'] == 'C2']
if c2: params['C2'] = max(r['median'] for r in c2)

for pid in ['C4_READ', 'C4_WRITE']:
    items = [r for r in results if r['param_id'] == pid]
    if items: params[pid] = items[0]['median']

# M1-M8
for pid in ['M1', 'M2']:
    items = [r for r in results if r['param_id'] == pid]
    if items: params[pid] = max(r['median'] for r in items)

for pid in ['M3', 'M4', 'M5']:
    items = [r for r in results if r['param_id'] == pid]
    if items: params[pid] = items[0]['median']

m6 = [r for r in results if r['param_id'] == 'M6']
if m6: params['M6'] = max(r['median'] for r in m6)

m8 = [r for r in results if r['param_id'] == 'M8' and r['param_name'] == 'MTE启动开销']
if m8: params['M8'] = m8[0]['value']

with open('results/extracted_params.json', 'w') as f:
    json.dump(params, f, indent=2)
print(f'  ✅ 提取了 {len(params)} 个参数')
" 2>&1 | grep -v -E "UserWarning|SyntaxWarning|warnings.warn"

echo ""
echo "[2/3] 生成 C5 折线图..."
python3 -c "
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('c5_scaling.csv')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('C5: MatMul Latency Scaling on Ascend 310P3', fontsize=14)

for ax, (mask, title, color) in zip(axes.flat, [
    ((df['n']==256)&(df['k']==256), 'Varying M (N=K=256)', 'b'),
    ((df['m']==256)&(df['k']==256), 'Varying N (M=K=256)', 'r'),
    ((df['m']==256)&(df['n']==256), 'Varying K (M=N=256)', 'g'),
    ((df['m']==df['n'])&(df['n']==df['k']), 'Square (NxNxN)', 'm'),
]):
    sub = df[mask].sort_values(df.columns[0])
    ax.plot(sub.iloc[:,0], sub['latency_ms'], f'{color}o-', linewidth=2, markersize=8)
    ax.set_xlabel(sub.columns[0]); ax.set_ylabel('Latency (ms)')
    ax.set_title(title); ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    if title.startswith('Square'): ax.set_yscale('log')

plt.tight_layout()
plt.savefig('results/c5_scaling_chart.png', dpi=150, bbox_inches='tight')
print('  ✅ C5 折线图已保存')
" 2>&1 | grep -v -E "UserWarning|SyntaxWarning|warnings.warn"

echo ""
echo "[3/3] 检查 HTML 仪表板..."
if [ -f results/dashboard.html ]; then
    echo "  ✅ HTML 仪表板已存在: results/dashboard.html"
else
    echo "  ⚠️  HTML 仪表板不存在，请手动创建"
fi

echo ""
echo "============================================================"
echo "  报告文件:"
echo "    results/extracted_params.json  — 关键参数"
echo "    results/c5_scaling_chart.png   — C5 折线图"
echo "    results/dashboard.html         — HTML 仪表板"
echo "    report_template.md             — 完整实验报告"
echo "============================================================"
