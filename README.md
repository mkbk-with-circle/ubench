# ubench — 昇腾 910B 微基准测试（Lab2）

并行与分布式计算导论 Lab2：在昇腾 NPU 上测量 DaVinci 微架构参数（S1–S3, V1–V5, C1–C5, M1–M8）。

## 目录结构

```
ubench/
├── common/          # 计时与统计工具
├── scalar/          # S1–S3
├── vector/          # V1–V5
├── cube/            # C1–C5
├── mte/             # M1–M8
├── run_all.py       # 批量运行
└── report_template.md
```

## 环境（卓越中心 cntrain / SCOW HPC）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 若用 PyTorch 路径：
pip install torch-npu  # 按集群 CANN 版本匹配
```

## 运行

```bash
# 全部
python run_all.py

# 按类别
python run_all.py --category vector
python run_all.py --quick   # 快速试跑
```

结果默认写入 `results/`（已 gitignore）。

## 集群同步

```bash
git clone https://github.com/mkbk-with-circle/ubench.git
cd ubench
python run_all.py --category scalar
```
