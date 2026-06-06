#!/usr/bin/env bash
# ============================================================
#  昇腾 NPU μbench 全量运行脚本
#  用法: bash scripts/run_all.sh [--quick] [--category scalar|vector|cube|mte|all]
#
#  示例:
#    bash scripts/run_all.sh                    # 运行全部
#    bash scripts/run_all.sh --quick            # 快速模式
#    bash scripts/run_all.sh --category cube    # 只跑 Cube
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "  昇腾 NPU μbench 全量运行"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# 激活环境（直接 source CANN 确保 LD_LIBRARY_PATH 传递到子进程）
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
echo "✅ CANN 环境已激活"

# 解析参数
QUICK=""
CATEGORY="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) QUICK="--quick"; shift ;;
        --category) CATEGORY="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo ""
echo "运行模式: category=$CATEGORY, quick=$QUICK"
echo ""

cd "$PROJECT_DIR"

# 运行测试，带超时保护
TIMEOUT=1800  # 30 分钟超时
START_TIME=$(date +%s)

python3 run_all.py --category $CATEGORY $QUICK 2>&1 | grep -v -E "UserWarning|SyntaxWarning|warnings.warn|dirpath|\"\"\"" &
PID=$!

# 轮询监控
echo "监控进程 PID=$PID，超时=${TIMEOUT}s"
echo ""
while kill -0 $PID 2>/dev/null; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [ $ELAPSED -gt $TIMEOUT ]; then
        echo ""
        echo "❌ 超时（${TIMEOUT}s），终止进程..."
        kill $PID 2>/dev/null
        exit 1
    fi
    sleep 30
    echo "[$(date +%H:%M:%S)] 运行中... (${ELAPSED}s)"
done

wait $PID
EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ 运行完成！耗时: ${ELAPSED}s"
    echo "  结果目录: $PROJECT_DIR/results/"
    echo ""
    ls -lh "$PROJECT_DIR/results/"
else
    echo "  ❌ 运行失败，退出码: $EXIT_CODE"
fi
echo "============================================================"
