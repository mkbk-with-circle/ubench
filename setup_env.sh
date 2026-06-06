#!/usr/bin/env bash
# ============================================================
#  Ascend NPU μbench 环境快速恢复脚本
#  用法: source setup_env.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/3] 激活 CANN 环境..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null

echo "[2/3] 检查 Python 依赖..."
python3 -c "import torch; import torch_npu; import numpy; import yaml" 2>/dev/null || {
    echo "  缺少依赖，正在安装..."
    pip3 install torch==2.6.0 torch-npu==2.6.0.post5 pyyaml numpy 2>/dev/null
}

echo "[3/3] 验证 NPU..."
python3 -c "
import torch, torch_npu
assert torch.npu.is_available(), 'NPU 不可用'
print(f'  torch={torch.__version__}, torch_npu={torch_npu.__version__}')
print(f'  NPU: {torch.npu.get_device_name(0)} × {torch.npu.device_count()}')
" 2>&1 | grep -v UserWarning

echo ""
echo "环境就绪。运行测试:"
echo "  cd $SCRIPT_DIR"
echo "  python3 run_all.py"
