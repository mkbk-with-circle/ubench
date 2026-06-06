#!/usr/bin/env bash
# ============================================================
#  昇腾 NPU μbench 环境快速恢复脚本
#  用法: source scripts/setup_env.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "  昇腾 NPU μbench 环境配置"
echo "============================================================"

# 1. 激活 CANN 环境
echo ""
echo "[1/4] 激活 CANN 环境..."
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
    echo "  ✅ CANN 环境已激活"
else
    echo "  ❌ 未找到 CANN toolkit，请先安装"
    exit 1
fi

# 2. 检查 Python 版本
echo ""
echo "[2/4] 检查 Python 环境..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python: $PYTHON_VERSION"

# 3. 安装 Python 依赖
echo ""
echo "[3/4] 检查 Python 依赖..."
python3 -c "import torch; import torch_npu; import numpy; import yaml; import scipy; import pandas; import matplotlib" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✅ 所有依赖已安装"
else
    echo "  ⚠️  缺少依赖，正在安装..."
    pip3 install torch==2.6.0 torch-npu==2.6.0.post5 "numpy<2.0" pyyaml scipy decorator attrs absl-py psutil pandas matplotlib 2>/dev/null
    echo "  ✅ 依赖安装完成"
fi

# 4. 验证 NPU
echo ""
echo "[4/4] 验证 NPU 设备..."
python3 -c "
import torch, torch_npu
assert torch.npu.is_available(), 'NPU 不可用'
print(f'  ✅ NPU: {torch.npu.get_device_name(0)} × {torch.npu.device_count()}')
print(f'  torch: {torch.__version__}')
print(f'  torch_npu: {torch_npu.__version__}')
" 2>&1 | grep -v -E "UserWarning|warnings.warn"

echo ""
echo "============================================================"
echo "  环境就绪！运行测试:"
echo "    cd $PROJECT_DIR"
echo "    bash scripts/run_all.sh"
echo "============================================================"
