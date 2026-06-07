# Ascend 310P3 微基准测试结果

> 测试环境: CANN 8.2.RC1 · Ascend 310P3 (4 NPU) · aarch64-linux · 2026-06-07

## 测试结果总览

| 类别 | 基准测试 | 状态 | 调整后斜率 (μs/op) | 原始斜率 (μs/op) | R² |
|------|---------|------|-------------------|------------------|-----|
| MTE | copy_bw | ✅ 通过 | 0.1236 | 0.1350 | 0.99999 |
| MTE | startup_latency | ✅ 通过 | 0.2746 | 0.2846 | 0.99997 |
| MTE | granularity | ✅ 通过 | 0.1454 | 0.1441 | 0.99919 |
| Vector | add_latency | ✅ 通过 | 0.0224 | 0.0335 | 0.99983 |
| Vector | mul_latency | ✅ 通过 | 0.0236 | 0.0325 | 0.99905 |
| Vector | throughput | ✅ 通过 | 0.1142 | 0.1253 | 0.99999 |
| Vector | pipeline_depth | ✅ 通过 | 0.0228 | 0.0338 | 0.99983 |
| Scalar | arith_latency | ✅ 通过 | 0.0011 | 0.0055 | 0.93214 |
| Scalar | branch_overhead | ✅ 通过 | 0.0107 | 0.0106 | 0.99925 |
| Cube | tile_latency | ⏭ 跳过 | — | — | — |
| Cube | throughput | ⏭ 跳过 | — | — | — |
| Cube | scaling | ⏭ 跳过 | — | — | — |

**9/12 基准测试通过**，覆盖 MTE、Vector、Scalar 三大类微架构参数。Cube 在 310P 上因 `Matmul` 模板兼容性问题挂起，已跳过。

---

## 报告用派生指标

> 以下派生值基于 kernel 源码常量推导，斜率本身可信（R² 均 >0.99，scalar_arith 除外）。

| 参数 | 调整后斜率 | 派生值 | 说明 |
|------|-----------|--------|------|
| MTE copy (M1) | 0.1236 μs/op | **~66 GB/s** | GM↔UB 1024B×2×4 blocks；每 repeat 搬运 8192B |
| MTE startup (M8) | 0.2746 μs/op | **275 ns** | 32B 完整 DataCopy 往返，非纯 DMA 启动开销 |
| MTE granularity | 0.1454 μs/op | **145 ns** | mode=0 最小 32B 粒度，非 Lab2 M7 容量曲线 |
| V3 throughput | 0.1142 μs/op | **~36 GFLOPS** | 4 blocks×4 lanes×256 FP32 Add = 4096 ops/repeat |
| V3 ops/cycle | — | **~0.035** | 4 blocks×4 lanes / (0.1142μs×1GHz) @1GHz |
| V1 add_latency | 0.0224 μs/op | **22.4 ns** | 含 PipeBarrier，310P 参考值 |
| V2 mul_latency | 0.0236 μs/op | **23.6 ns** | 含 PipeBarrier，310P 参考值 |
| V4 pipeline_depth | 0.0228 μs/op | **22.8 ns** | gap ops 被管线隐藏，与 V1 接近 |
| S1 arith_latency | 0.0011 μs/op | **1.14 ns** | ⚠ **不采用**: R²=0.932<0.95，baseline 与 target 成本接近 |
| S2 branch_overhead | 0.0107 μs/op | **10.7 ns** | 分支探针，R²=0.999 可信；非 S3 访存延迟 |
| Cube (3 项) | — | **N/A** | 310P Matmul 模板挂起，已跳过；待 910B 验证 |

### 派生公式

```
MTE 带宽  = blocks × kTileBytes × 2 / slope_us
          = 4 × 1024 × 2 / 0.1236μs ≈ 66 GB/s

Vector GFLOPS = blocks × lanes × elems / slope_us / 1e9
              = 4 × 4 × 256 / 0.1142μs / 1e9 ≈ 36 GFLOPS

Vector ops/cycle = blocks × lanes / (slope_us × freq_MHz)
                 = 4 × 4 / (0.1142μs × 1000) ≈ 0.035 @1GHz
```

### 310P 免责声明

- 本结果 **不能** 与 Lab2 910B 助教标定值直接对比（不同硬件、不同 CANN 版本）
- Lab 参数映射为**近似**：
  - M1 实为 GM↔UB 带宽探针，非纯 L2→L1
  - M8 实为 DataCopy 启动+往返延迟，非纯 DMA 启动开销
  - S1 标量算术 R²<0.95，标记为不可靠/实验性，不作为正式 KPI
- `scalar_arith_latency` 中 repeat=500 时 adjusted 为负值（-0.58μs），说明 baseline 开销已超过 target，该点无效
- Cube 在 310P 上不可用，需在 910B 上重新测量

---

## 本次所做的更改

### 1. 构建系统修复

原始 CMakeLists.txt 无法在当前环境 (CANN 8.2.RC1) 下编译，原因：
- CANN 安装路径检测失败 (`ascendc_devkit` 目录不存在)
- Ascend C cmake 流程的 ExternalProject 无法正确传递 ACL 头文件路径
- 内核源码中的未命名参数导致 host stub 生成器解析失败

**修复方案**:
- 创建 `cann_wrapper/` 目录，通过符号链接模拟标准 CANN 目录结构
- 创建 `local_ascendc_cmake/` 目录，修补 `device_precompile_project/CMakeLists.txt` 添加 ACL 头文件路径
- 修复内核源码中的未命名参数（如 `uint32_t)` → `uint32_t mode)`）

### 2. 内核源码兼容性修复

**问题**: DaVinci AI Core 不支持 `static_cast<float>(uint32_t)` 类型转换。

**修复**: 使用 union 技巧替代：
```cpp
// 原始代码 (编译失败)
acc.SetValue(0, static_cast<float>(i));

// 修复后
{ union { uint32_t u; float f; } c; c.u = i; acc.SetValue(0, c.f); }
```

> **注意**: 以上修改仅影响 baseline 分支中的可观测占位操作（`if ((i & 0x7ffu) == 0)` 分支），不改变 target kernel 的 `Process()` 运算逻辑。

### 3. 环境配置脚本

- `setup_env.sh`: 一键环境配置
- `scripts/build_one_click.sh`: 一键构建（包含所有修补步骤）
- `scripts/run_all.sh`: 支持 `SKIP_CUBE=1` 自动跳过 Cube 基准

### 4. 可视化

- `results/html/index.html`: 完全离线单文件交互式页面
  - `EMBEDDED_DATA` JSON 块内嵌所有数据和 kernel 常量
  - `deriveMetric` 从 kernel 常量推导派生指标，不使用 `size_bytes`
  - scalar_arith_latency 低可信度警告
  - Cube 显示「310P 不支持 / 已跳过」
- `results/html/build_data.py`: 从 CSV 重新生成 EMBEDDED_DATA

---

## Cube 基准测试无法完成的原因

### 现象

三个 Cube 基准测试 (`cube_tile_latency`, `cube_throughput`, `cube_scaling`) 在运行时挂起：
- 进程状态: `S (sleeping)`
- 卡在内核态: `logic_cq_wait_event` 系统调用
- 无论参数如何调整（blocks=1, repeats=1, size=1024），均无法完成

### 根本原因

**`lib/matmul_intf.h` 中的 `Matmul` 模板在 310P3 (m200 架构) 上存在兼容性问题。**

具体分析：

#### 1. v200 代码路径的缓存维护问题

`IterateAll()` 方法内部的 v200 代码路径（`__CCE_AICORE__ == 200`）：
```cpp
#if __CCE_AICORE__ == 200
    GlobalTensor<uint64_t> global;
    global.SetGlobalBuffer((__gm__ uint64_t*)0);
    DataCacheCleanAndInvalid<uint64_t, CacheLine::ENTIRE_DATA_CACHE>(global);
#endif
```
这行代码对地址 0 进行全缓存行的 Clean+Invalid 操作，在 310P3 上导致硬件挂起。

#### 2. 调度器初始化问题

即使通过 `Init()` 方法传入正确的 `TCubeTiling` 参数，调度器仍无法正常初始化。

#### 3. 底层 API 也受影响

直接使用 `MmadImpl()` 函数（最底层 Cube 单元 API）也会挂起，说明问题在 Cube 单元的驱动层面。

### 处理方式

- 在 310P 上自动跳过 Cube 基准（`SKIP_CUBE=1`）
- 不修改任何 Cube 内核源码
- 待 910B 环境重新测量

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `<benchmark>.txt` | 基准测试的 stdout 输出（人类可读） |
| `<benchmark>.csv` | 原始 repeat sweep 数据 + 拟合斜率/截距/R² |
| `summary.csv` | 每个基准测试一行汇总 |
| `html/index.html` | 交互式可视化页面 (Chart.js)，完全离线单文件 |
| `html/build_data.py` | 从 CSV 生成 EMBEDDED_DATA JSON，`--embed` 写回 HTML |
| `msprof_<benchmark>/` | 可选的 msprof 性能分析输出 |

## 测试参数

```bash
--device 0      # NPU 设备 ID
--warmup 3      # 预热迭代次数
--iters 10      # 测量迭代次数
--blocks 4      # AI Core 块数
--repeats 500   # 基础 repeat 数（实际 sweep: 500, 1000, 2500, 5000）
--size 65536    # 缓冲区大小 (64KB)
```
