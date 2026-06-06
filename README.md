# ubench — 昇腾 310P3 微基准测试（Lab2）

并行与分布式计算导论 Lab2：在昇腾 NPU 上测量 DaVinci 微架构参数（S1–S3, V1–V5, C1–C5, M1–M8）。

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/mkbk-with-circle/ubench.git
cd ubench

# 2. 配置环境（每次登录新终端时执行）
source scripts/setup_env.sh

# 3. 一键运行全部测试 + 生成报告
bash scripts/run_and_report.sh
```

## 目录结构

```
ubench/
├── scripts/           # 🔧 运行脚本（从这里开始）
│   ├── setup_env.sh       # 环境配置
│   ├── run_all.sh         # 运行测试
│   ├── generate_report.sh # 生成报告
│   ├── run_and_report.sh  # 一键运行+报告
│   └── README.md          # 脚本说明
├── common/            # 计时与统计工具
├── scalar/            # S1–S3 标量单元
├── vector/            # V1–V5 向量单元
├── cube/              # C1–C5 矩阵乘单元
├── mte/               # M1–M8 数据搬运单元
├── results/           # 测量结果输出
│   ├── all_results.json       # 汇总结果
│   ├── extracted_params.json  # 关键参数
│   ├── c5_scaling_chart.png   # C5 折线图
│   └── dashboard.html         # HTML 仪表板
├── run_all.py         # Python 入口
├── report_template.md # 实验报告
└── setup_env.sh       # 环境恢复脚本
```

## 运行选项

```bash
# 运行全部
bash scripts/run_all.sh

# 快速模式
bash scripts/run_all.sh --quick

# 只运行特定类别
bash scripts/run_all.sh --category scalar   # S1-S3
bash scripts/run_all.sh --category vector   # V1-V5
bash scripts/run_all.sh --category cube     # C1-C5
bash scripts/run_all.sh --category mte      # M1-M8

# 生成报告（已有结果时）
bash scripts/generate_report.sh
```

## 环境要求

- 硬件：昇腾 NPU（310P3 / 910B）
- 系统：openEuler / CentOS / Ubuntu (aarch64)
- 软件：CANN 8.x, Python 3.10+

## 查看结果

```bash
# 打开 HTML 仪表板
open results/dashboard.html

# 查看关键参数
cat results/extracted_params.json

# 查看实验报告
cat report_template.md
```
