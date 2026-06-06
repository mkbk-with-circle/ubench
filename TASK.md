# Lab2 微基准测试任务路线图

## 目标
完成并行与分布式计算导论 Lab2：在昇腾 NPU 上测量 DaVinci 微架构参数（S1-S3, V1-V5, C1-C5, M1-M8），生成 20 页实验报告。

---

## 当前状态

### 代码层面：全部 21 个 μbench 已实现 ✅
| 分类 | 文件 | 状态 |
|------|------|------|
| Common | `benchmark.py` (框架), `utils.py` (工具) | ✅ |
| Scalar (S1-S3) | `s1_latency.py`, `s2_throughput.py`, `s3_mem_latency.py` | ✅ |
| Vector (V1-V5) | `v1_add_latency.py`, `v2_mul_latency.py`, `v3_throughput.py`, `v4_pipeline_depth.py`, `v5_reg_latency.py` | ✅ |
| Cube (C1-C5) | `c1_tile_latency.py`, `c2_throughput.py`, `c3_pipeline_depth.py`, `c4_l0_latency.py`, `c5_scaling.py` | ✅ |
| MTE (M1-M8) | `m1_l1_read_bw.py`, `m2_l1_write_bw.py`, `m3_m4_m5_l0_bw.py`, `m6_hbm_latency.py`, `m7_buffer_capacity.py`, `m8_startup_overhead.py` | ✅ |

### 结果层面：部分已有
| 文件 | 内容 |
|------|------|
| `results/scalar_results.json` | ✅ S1-S3 已有数据 |
| `results/vector_results.json` | ✅ V1-V5 已有数据 |
| `results/cube_results.json` | ❌ 缺失 |
| `results/mte_results.json` | ❌ 缺失 |

### 环境层面：未就绪
- `torch` 和 `torch_npu` 未安装
- 当前硬件：Ascend 310P3（代码部分硬编码 910B 参数，需适配）
- 之前的 scalar/vector 结果说明曾在另一环境跑通

---

## Road Map

### Phase 1: 环境搭建（~30min）
- [ ] Step 1.1: 安装 torch + torch_npu（匹配 Ascend 310P3 / CANN 版本）
- [ ] Step 1.2: 运行 `test_imports.py` 验证所有模块可导入
- [ ] Step 1.3: 写环境恢复脚本 `setup_env.sh`（一键 source + pip install）

### Phase 2: 运行基准测试（~1-2h）
- [ ] Step 2.1: 跑 Scalar (S1-S3) — 快速验证 NPU 可用，~5min
- [ ] Step 2.2: 跑 Vector (V1-V5) — ~10min
- [ ] Step 2.3: 跑 Cube (C1-C5) — ~20min（含 C5 大矩阵）
- [ ] Step 2.4: 跑 MTE (M1-M8) — ~15min
- [ ] Step 2.5: 遇报错则逐步修复代码适配 310P3
- [ ] Step 2.6: 全部跑通后运行 `run_all.py` 生成汇总 `results/all_results.json`

### Phase 3: 结果整理（~30min）
- [ ] Step 3.1: 从 4 个 JSON 中提取 21 个参数，填入 Figure 1 表格
- [ ] Step 3.2: 用 C5 数据生成折线图（matplotlib，延迟 vs 矩阵规模）
- [ ] Step 3.3: 整理所有原始数据，准备附录

### Phase 4: 撰写报告（~1-2h）
- [ ] Step 4.1: 填写 `report_template.md` 所有空白字段
- [ ] Step 4.2: 分析各参数合理性，与理论值对比（Section 4.1）
- [ ] Step 4.3: 讨论误差来源、DaVinci 架构特征（Section 4.2）
- [ ] Step 4.4: 撰写总结（Section 5）
- [ ] Step 4.5: 导出最终 PDF

---

## 注意事项
1. 关注执行命令是否卡死，每分钟询问一次，防止卡死
2. 尽可能满足 lab2.pdf 的要求
3. 不成功不要停止
4. 配置好环境后写好快速恢复脚本
5. 当前硬件是 310P3，与 910B 有差异，理论值需相应调整
