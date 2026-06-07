# Ascend C ubench

This is a clean Ascend C + C++/ACL rewrite of the previous Python/PyTorch
ubench. The old Python implementation launched PyTorch tensor ops inside the
measured loop, so it measured framework dispatch and synchronization overhead
instead of DaVinci AI Core microarchitecture parameters.

## Layout

```text
ubench_ascendc/
├── common/              # ACL runtime helpers, fitting, CSV output
├── mte/                 # DataCopy bandwidth, startup latency, granularity
├── vector/              # Add/mul latency, throughput, pipeline-depth probes
├── cube/                # Matmul/MMAD tile, throughput, scaling probes
├── scalar/              # Experimental scalar probes, not ground truth by default
├── scripts/             # build/run/profile/report scripts
└── results/             # CSV and text output
```

Each benchmark follows the CUDA ubench style: a small C++ host runner allocates
device memory, launches one Ascend C kernel, synchronizes, reads back an
observable result, and prints CSV-friendly output. The target loop is inside the
`__global__ __aicore__` kernel.

## Build

Set CANN environment first:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_SOC_VERSION=Ascend310P3
export ASCEND_OPT_LEVEL=O2
```

Build host runners:

```bash
make build
```

Build/link device kernels with the CANN Kernel Launch flow used by the local
CANN installation:

```bash
make kernels
```

`common/launch_stubs.cpp` provides weak symbols that intentionally fail at
runtime if real Ascend C launch stubs are missing. This prevents accidentally
trusting host-only binaries.

## Run

```bash
make run
```

Common environment variables:

```bash
DEVICE_ID=0 WARMUP=5 ITERS=20 BLOCKS=8 REPEATS=1000 SIZE_BYTES=1048576 make run
```

Per-benchmark profiling is optional:

```bash
make profile BENCH=mte_copy_bw
```

This wraps `msprof op` and stores output below `results/msprof_<bench>/`.

## Measurement Discipline

The project does not rely on undocumented DaVinci inline assembly. Instead it
uses public Ascend C APIs and validates that the compiler did not invalidate the
benchmark.

Rules:

- Use `-O2` or `-O3` for performance numbers. Use `-O0` only for debugging.
- Keep the measured loop on the device side.
- The measured loop should contain only the target operation whenever possible.
- Write back only one final output/checksum after the loop.
- Pair each target kernel with a baseline kernel that preserves loop/control
  structure while removing the target operation.
- Sweep repeat counts and fit:

```text
time(N) = fixed_overhead + N * per_op_cost
```

The fitted slope is the reported per-operation cost. The intercept captures
fixed launch, writeback, and synchronization costs. Low R2 means the benchmark is
not clean enough to trust.

Use `PipeBarrier`, `SetFlag`, and `WaitFlag` only when the benchmark semantics
require pipeline ordering. Do not add synchronization to a raw throughput test
unless that synchronization is part of the parameter being measured.

## Current Kernel Coverage

- `mte/copy_bw`: GM to UB to GM copy bandwidth.
- `mte/startup_latency`: repeated 32B DataCopy startup probe.
- `mte/granularity`: DataCopy size sweep in 32B units.
- `vector/add_latency`: RAW dependent FP32 vector add chain.
- `vector/mul_latency`: RAW dependent FP32 vector multiply chain.
- `vector/throughput`: independent FP32 vector add lanes.
- `vector/pipeline_depth`: dependent vector add plus independent gap ops.
- `cube/*`: Matmul/MMAD template based on Ascend C `lib/matmul_intf.h`.
- `scalar/*`: experimental scalar probes; treat as non-ground-truth until
  verified against compiler output and profiling.

## Outputs

Each benchmark prints and writes:

- raw total time
- baseline time
- adjusted time
- raw fitted slope/intercept/R2
- baseline-adjusted fitted slope/intercept/R2
- output checksum

`scripts/summarize_csv.sh` creates `results/summary.csv` from individual CSVs.
