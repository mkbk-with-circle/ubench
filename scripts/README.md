# 昇腾 NPU μbench 脚本目录

## 快速开始

```bash
# 1. 配置环境（每次登录新终端时执行）
source scripts/setup_env.sh

# 2. 一键运行全部测试 + 生成报告
bash scripts/run_and_report.sh

# 3. 查看结果
open results/dashboard.html
```

## 脚本说明

| 脚本 | 功能 | 用法 |
|------|------|------|
| `setup_env.sh` | 配置 CANN + Python 环境 | `source scripts/setup_env.sh` |
| `run_all.sh` | 运行基准测试 | `bash scripts/run_all.sh [--quick] [--category X]` |
| `generate_report.sh` | 生成可视化报告 | `bash scripts/generate_report.sh` |
| `run_and_report.sh` | 一键运行+报告 | `bash scripts/run_and_report.sh` |

## 参数说明

```bash
# 快速模式（减少迭代次数）
bash scripts/run_all.sh --quick

# 只运行特定类别
bash scripts/run_all.sh --category scalar   # S1-S3
bash scripts/run_all.sh --category vector   # V1-V5
bash scripts/run_all.sh --category cube     # C1-C5
bash scripts/run_all.sh --category mte      # M1-M8

# 组合使用
bash scripts/run_all.sh --quick --category cube
```

## 输出文件

运行完成后，`results/` 目录下包含：

| 文件 | 内容 |
|------|------|
| `all_results.json` | 全部 216 个测量结果汇总 |
| `scalar_results.json` | S1-S3 原始数据 |
| `vector_results.json` | V1-V5 原始数据 |
| `cube_results.json` | C1-C5 原始数据 |
| `mte_results.json` | M1-M8 原始数据 |
| `extracted_params.json` | 22 个关键参数提取 |
| `c5_scaling_chart.png` | C5 折线图 |
| `dashboard.html` | HTML 可视化仪表板 |

## 环境要求

- 硬件：昇腾 NPU（310P3 / 910B）
- 系统：openEuler / CentOS / Ubuntu (aarch64)
- 软件：CANN 8.x, Python 3.10+
- 依赖：torch 2.6.0, torch-npu 2.6.0.post5

## 常见问题

**Q: 首次运行很慢？**
A: CANN 首次编译 kernel 需要较长时间（数十秒），后续调用会快很多。

**Q: 报 `libhccl.so` 错误？**
A: 确保已执行 `source scripts/setup_env.sh` 激活 CANN 环境。

**Q: 报 `np.float_` 错误？**
A: NumPy 2.0 与 CANN 不兼容，需降级：`pip3 install "numpy<2.0"`

**Q: 如何在其他机器上复现？**
A: 克隆仓库后执行：
```bash
git clone https://github.com/mkbk-with-circle/ubench.git
cd ubench
source scripts/setup_env.sh
bash scripts/run_and_report.sh
```
