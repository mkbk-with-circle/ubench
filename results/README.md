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
| Cube | tile_latency | ❌ 超时 | — | — | — |
| Cube | throughput | ❌ 超时 | — | — | — |
| Cube | scaling | ❌ 超时 | — | — | — |

**9/12 基准测试通过**，覆盖 MTE、Vector、Scalar 三大类微架构参数。

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

### 3. 环境配置脚本

- `setup_env.sh`: 一键环境配置
- `scripts/build_one_click.sh`: 一键构建（包含所有修补步骤）

### 4. 可视化

- `results/html/index.html`: 使用 Chart.js 的交互式可视化页面
  - 每个类别的 time(N) 线性拟合图
  - 原始斜率 vs 调整后斜率对比
  - 拟合质量 R² 对比
  - 综合参数对比（对数刻度）

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

即使通过 `Init()` 方法传入正确的 `TCubeTiling` 参数：
```cpp
TCubeTiling tiling;
tiling.usedCoreNum = 1;
tiling.M = 16; tiling.N = 16; tiling.Ka = 16;
// ... 其他字段
mm_.Init(&tiling);
```
调度器 (`MATMUL_MODULE(Scheduler)`) 仍无法正常初始化，`Iterate()` 方法返回后进入死循环。

#### 3. 底层 API 也受影响

直接使用 `MmadImpl()` 函数（最底层 Cube 单元 API）也会挂起：
```cpp
MmadParams params;
params.m = 16; params.n = 16; params.k = 16;
MmadImpl(c, a, b, params);  // 挂起
```
说明问题不在 `Matmul` 模板封装层，而是与 310P3 的 Cube 单元驱动交互有关。

### 已尝试的修复

| 方案 | 结果 |
|------|------|
| 添加 `TCubeTiling` 初始化 | ❌ 仍挂起 |
| 调整矩阵维度 (16×16×16) | ❌ 仍挂起 |
| 使用 `MmadImpl` 替代 `Matmul` 模板 | ❌ 仍挂起 |
| 使用不同的 `QuePosition` | ❌ 仍挂起 |
| 尝试不同 NPU 设备 (device 0~3) | ❌ 均挂起 |

### 建议解决方案

1. **升级 CANN 版本**: 当前 8.2.RC1 可能对 310P3 的 Cube 单元支持不完善，后续版本可能修复
2. **使用 ACT 库**: Ascend Computing Template 库 (`ascendc/act/`) 提供了更高级的 `DeviceMatmul` API，自动处理 tiling 和调度
3. **联系华为技术支持**: 确认 310P3 上 `Matmul` 模板的正确使用方式，以及 `DataCacheCleanAndInvalid` 对地址 0 的行为

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `<benchmark>.txt` | 基准测试的 stdout 输出（人类可读） |
| `<benchmark>.csv` | 原始 repeat sweep 数据 + 拟合斜率/截距/R² |
| `summary.csv` | 每个基准测试一行汇总 |
| `html/index.html` | 交互式可视化页面 (Chart.js) |
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
