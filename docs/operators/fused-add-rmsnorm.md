# Fused Add RMSNorm

Residual addition followed by RMS normalization, for the block norms and final
`norm_f` work item in [Nemotron roadmap #434](https://github.com/RL-Align/RL-Kernel/issues/434).
The target model is `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, whose roadmap
specifies hidden width **D = 2688**, FP32 norm computation/weights, and eps = 1e-5.

This implementation exposes an explicit **FP32-output** contract. The roadmap also
records `residual_in_fp32=false`; model integration must align residual/output
cast points with the maintainer before treating this as a drop-in strict model
implementation. Generic operator registration does not establish full-model parity.

## Definition and tensor contract

Flatten the leading dimensions into M rows. For each row i and feature j:

$$
u_{ij}=x_{ij}+r_{ij},\qquad
q_i=\left(\frac{1}{D}\sum_{k=0}^{D-1}u_{ik}^2+\epsilon\right)^{-1/2},\qquad
y_{ij}=(u_{ij}q_i)w_j.
$$

Forward returns `(y, updated_residual)`, where `updated_residual = u`.

| Direction | Tensor | Shape | Dtype / requirements |
| --- | --- | --- | --- |
| Input | `x` | `[..., D]` | FP16, BF16, or FP32; D > 0 |
| Input | `residual` | same as x | FP16, BF16, or FP32, independently of x |
| Input | `weight` | `[D]` | FP16, BF16, or FP32; model workload uses FP32 |
| Input | `eps` | scalar | finite positive Python value; default 1e-5; no gradient |
| Output | `y` | same as x | FP32 |
| Output | `updated_residual` | same as x | FP32 |

All tensors share a device. Noncontiguous inputs are accepted and copied to
contiguous storage in the Triton wrapper. Empty leading dimensions (M = 0) are
supported. The residual sum, row statistics, normalization, and weight multiply
use FP32 with no intermediate downcast. Inputs must stay within a numerical
range where FP32 residual addition and squared sums are finite; the wrapper
validates metadata and eps, without a device-wide scan of tensor values.

For backward, let g be the incoming gradient of y and h the incoming gradient of
the residual output. Define z = u q and a = g w, elementwise:

$$
c_i=\frac{1}{D}\sum_{j=0}^{D-1}a_{ij}z_{ij},\qquad
\frac{\partial L}{\partial x_{ij}}=
\frac{\partial L}{\partial r_{ij}}=q_i(a_{ij}-z_{ij}c_i)+h_{ij},\qquad
\frac{\partial L}{\partial w_j}=\sum_{i=0}^{M-1}g_{ij}z_{ij}.
$$

| Direction | Tensor | Shape | Dtype / requirements |
| --- | --- | --- | --- |
| Upstream | `grad_y`, `grad_updated_residual_output` | same as x | FP32 from the public FP32 outputs; an unused branch contributes zero |
| Saved | `updated_residual`, `inverse_rms`, `weight` | `[..., D]`, `[M]`, `[D]` | FP32 sum/statistic; original weight dtype |
| Returned | `grad_x`, `grad_residual` | same as x | cast once to each corresponding input dtype |
| Returned | `grad_weight` | `[D]` | cast once to the weight dtype |

Backward arithmetic and the weight-gradient workspace are FP32. Triton supports
first-order autograd; higher-order derivatives are not supported. PyTorch's
reference uses ordinary autograd. Do not modify the returned residual in place
before backward: it is also a saved forward value protected by autograd's version
check.

## Entry points and dispatch

```python
import torch
from rl_engine.kernels.registry import KernelRegistry

x = torch.randn(2, 16, 2688, device="cuda", dtype=torch.bfloat16, requires_grad=True)
residual = torch.randn_like(x, requires_grad=True)
weight = torch.ones(2688, device=x.device, dtype=torch.float32, requires_grad=True)
op = KernelRegistry().get_op("fused_add_rmsnorm", device=x.device)
y, updated_residual = op(x, residual, weight, eps=1e-5)
(y.square().mean() + updated_residual.square().mean()).backward()
```

| Backend | Implementation | Dispatch |
| --- | --- | --- |
| CUDA | `TritonFusedAddRMSNormOp` | preferred; H100 kernel/configuration tests passed; conservative automatic weight-reduction selection |
| ROCm | same Triton implementation via `torch.cuda` | preferred; ROCm validation pending |
| PyTorch | `NativeFusedAddRMSNormOp` | CPU reference and fallback when the Triton backend cannot load |

CPU/MUSA/NPU registry entries use the native implementation; this does not claim
accelerator validation for MUSA/NPU. A successfully loaded Triton Op does not
silently retry through PyTorch if its input validation or kernel launch fails.
Both implementations are exported from their respective `norm` packages and
own their validation and dtype constants independently.

Direct Triton calls accept `cuda`, `hip`, `xpu`, and `musa` device types, following
the existing operators. Execution requires a compatible Triton backend; XPU/MUSA
validation remains pending. Standard ROCm PyTorch uses the `cuda` device type.

## Kernel design and consistency

- Forward uses one program per row and `next_power_of_2(D)` lanes, masking the
  padding to zero. D = 2688 uses a 4096-element tile. The row statistic is saved
  in FP32; the existing residual output is reused by backward.
- Backward reuses those saved values, computes input gradients and FP32 per-row
  weight contributions, then launches a separate weight reduction. CUDA/ROCm
  launches use the input GPU's device context and current stream; other accepted
  device types do not enter the CUDA context.
- Four warps and `enable_fp_fusion=False` fix each row's arithmetic independently
  of batch size. Grad-enabled, `no_grad`, and `inference_mode` forward use the
  same kernel. Tests compare raw bytes, including reordered/subset rows, for
  outputs, saved statistics, and corresponding input gradients.
- Explicit `RMSNormWeightGradStrategy.SEQUENTIAL`: each program owns 128
  features and folds row contributions in ascending order, without atomics.
- Explicit `RMSNormWeightGradStrategy.TILED` without a config: each program accumulates
  32 row lanes by 128 features, then reduces the row lanes once. It changes the
  FP32 addition order.
- Explicit `RMSNormWeightGradStrategy.PARALLEL` without a config: independent programs
  reduce fixed blocks of 256 rows by 128 features, write FP32 partials, and a
  second kernel merges them using the existing fixed 32-row-lane reduction.
  There are no floating-point atomics or locks. All launches use the same stream;
  no host synchronization is inserted. The partition depends on the explicit
  tile configuration, not the GPU's SM count or runtime scheduling.
- Experimental `RMSNormWeightGradStrategy.FUSED`: one program computes input
  gradients for a fixed group of rows and accumulates its weight contribution
  in registers. It writes only `[ceil(M / group_rows), D]` FP32 partials, followed
  by a fixed 32-lane merge. The default group has 64 rows; only this merge uses
  `block_cols` and `num_warps`. The input-gradient kernel keeps four warps and the
  original row arithmetic. Forward is unchanged. There are no atomic additions.
  This follows the grouped-accumulation idea in
  [Mamba's backward](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layer_norm.py),
  but groups use fixed row counts rather than the device's SM count, and the
  operator's FP32 outputs/saved statistics/cast points are retained. Large widths
  can increase register pressure; GPU correctness and performance validation are
  pending. FUSED is explicit-only and is not selected by the automatic policy.
- `RMSNormWeightGradConfig` exposes `block_rows`, `block_cols` and `num_warps` for
  experiments. Settings are saved per autograd call; changing a reusable Op
  afterwards does not change a graph's backward. Forward and per-row backward
  keep their original tile and four-warps configuration.

FUSED removes the `[M, D]` contribution workspace. At `M=65536, D=2688`, that
buffer is 672 MiB; fixed groups of 64 need 10.5 MiB of partials instead. These
are calculated scratch sizes, not total peak allocation or measured speedups.
Weight-gradient rounding can differ between grouping strategies. Repeated calls
with a fixed configuration must reproduce all gradients bitwise; different
groupings and separately reduced microbatches need not yield identical weight
gradient bytes.

At M = 8192, D = 2688, the original reductions launch only 21 programs, each
processing all 8192 rows. Default PARALLEL launches 672 partial programs followed
by 21 merge programs. Its additional partial buffer is only 32 x 2688 FP32 values
(0.328 MiB), on top of the existing 84 MiB per-row contributions. This addresses
limited parallelism, at the cost of an additional launch and workspace. Small
inputs may favor SEQUENTIAL or TILED. Automatic calls use the measured configurations
and conservative rules below, rather than these original explicit-strategy defaults.

Override selection explicitly:

```python
from rl_engine.kernels.ops.triton.norm import (
    RMSNormWeightGradConfig,
    RMSNormWeightGradStrategy,
    TritonFusedAddRMSNormOp,
)

op = TritonFusedAddRMSNormOp(
    weight_grad_strategy=RMSNormWeightGradStrategy.PARALLEL,
    weight_grad_config=RMSNormWeightGradConfig(block_rows=256, block_cols=128, num_warps=4),
)
```

SEQUENTIAL, TILED, and PARALLEL share the same forward and per-row backward kernels.
FUSED shares forward and preserves the per-row backward expressions in a grouped kernel.
Weight gradients sum across rows and are checked numerically plus repeatably for an identical call;
they are not promised to match bitwise across strategies, row reorderings, or
separately reduced microbatches. Full-model, distributed, and CUDA-to-ROCm parity
require separate validation. FP32 intermediates alone do not prove these properties.

### Automatic weight reduction

`TritonFusedAddRMSNormOp()` uses `weight_grad_strategy=None` to select a fixed
strategy/configuration from input metadata. M is the product of the leading
dimensions, not just the first dimension. Three supported input dtypes share
the following conservative defaults:

| Width D | Rows M (inclusive) | Strategy | block_rows | block_cols | num_warps |
| --- | --- | --- | ---: | ---: | ---: |
| any | 0..8 | SEQUENTIAL | 1 | 128 | 4 |
| any | 9..32 | TILED | 32 | 64 | 4 |
| 2688 | 33..16383 | TILED | 64 | 64 | 8 |
| 2688 | 16384 and above | PARALLEL | 512 | 64 | 4 |
| other widths | 33 and above | TILED | 64 | 64 | 8 |

These are broad policy choices, not measured optimal crossover points. H100
eager timings at small/medium M were too noisy to justify fine-grained rules.
The larger D=2688 cases support PARALLEL; the transition at 16384 remains a
conservative candidate for targeted regression. Other widths retain a one-launch
TILED reduction: the D=8193 probes did not show a consistent PARALLEL advantage.
Rows beyond the measured maximum 65536 follow the open-ended rules above; those
are explicit extrapolations, not additional measured results. No D rounding
or next-power-of-two bucketing broadens the D=2688 entry.

The table lives in the operator file as `WEIGHT_GRAD_POLICY`. Device-specific
entries take precedence over `"default"`; within a device tier, an exact input
dtype precedes the shared dtype rule, and an exact width precedes the fallback.
Ranges at the same device/dtype/width tier must not overlap. Until more devices
are tuned, they share the H100-informed defaults without a claim of equal
performance. Mixed input/weight dtypes remain supported, but the tuning workload
used matching x/residual dtypes and FP32 weights.

Selection caches up to 1024 metadata combinations. GPU names are separately
cached by backend and resolved device index; selection uses the input device,
not whichever GPU happens to be current. Device lookup errors propagate. There
is no timing/autotuning, tensor-value scan, synchronization, or graph-capture
dependent policy. Runtime edits to the static table require
`select_rmsnorm_weight_grad_plan.cache_clear()`.

Forward saves the resolved strategy/configuration on its own ctx. Changing an
Op later cannot change an existing graph's reduction order. All strategies
share the same forward; FUSED uses a grouped backward with the same per-row
expressions and four-warps setting. Automatic
selection may change weight-gradient addition order between different shapes;
it does not promise cross-strategy or microbatch weight-gradient byte equality.

An explicit strategy bypasses automatic selection and keeps its original default
configuration, shown above in the kernel design. An explicit config overrides
those settings. For backward compatibility, a config supplied without a strategy
keeps SEQUENTIAL rather than attaching arbitrary settings to an automatic strategy.

### Experimental grouped backward comparison

FUSED is not part of automatic selection while its GPU gates and performance are
unverified. Run correctness first, then compare complete public calls:

```bash
uv run --no-sync python -m pytest tests/nemotron/test_fused_add_rmsnorm*.py -q -rs -x && \
uv run --no-sync python benchmarks/benchmark_fused_add_rmsnorm.py \
  --preset fused --fused-configs --public-graph \
  --output-dir reports/fused-rmsnorm-grouped
```

The sparse preset covers 12 shapes across three input dtypes (36 cases), with
widths 129, 2688, 4096, 8192 and 8193 and model rows up to 65536. Eight providers
compare PyTorch, the automatic policy, SEQUENTIAL, the measured TILED/PARALLEL
controls, and FUSED groups of 16/64/256 rows. This produces 864 eager measurements,
or 1728 including `--public-graph`. Override `--rows`/`--cols` for a smaller smoke;
`--dry-run` prints counts without GPU execution. No standalone sum is measured
for FUSED because its accumulation is integrated into input-gradient computation.

Every candidate must pass reference output/gradient checks, training/inference
byte equality, row permutation/subset invariance, and (for Triton) repeatable
backward and byte equality of outputs/input gradients against the original row
path before any timing starts for that case. Graph returns are checked again
after timed replay. Reports record actual public peak allocation as well as
calculated contribution/partial workspace sizes. The comparison also includes
TILED/FUSED and PARALLEL/FUSED families; no measured winner is installed into
the policy automatically. This is a new experiment, not covered by earlier H100
results for the standalone reductions.

## Validation

The operator tests use an independent FP64 autograd reference, random upstreams
for both outputs, single-output losses, mixed dtypes, empty/strided inputs, unused
input gradients, repeated backward, saved-tensor immutability, masked tails, and
noncurrent-device launches on two GPUs. These checks cover the three original
strategies, automatic selection, and the explicit experimental FUSED strategy.
Configuration checks also cover partial/merge boundaries, poisoned
workspaces, guard regions, CUDA Graph replay and per-forward configuration retention. Default elementwise `(atol, rtol)` checks
are `(2e-5, 2e-5)` for FP32, `(3e-3, 3e-3)` for FP16, and `(2e-2, 2e-2)` for BF16.
The general harness additionally uses the shared `reduction` tolerance contract.

```bash
uv run --no-sync python -m pytest tests/nemotron/test_fused_add_rmsnorm*.py -q -rs
uv run --no-sync python scripts/check_operator.py \
  --op fused_add_rmsnorm --candidate triton --device cuda --dtype bf16 \
  --batch 2 --seq 16 --normalized-dim 2688 --eps 1e-5 --check-grad
```

A CPU run validates the native API, registry, general harness integration and
benchmark reporting. GPU skips are not evidence of Triton correctness or performance.

## H100 findings and optimization rationale

The H100 80GB HBM3 configuration sweep (PyTorch 2.13.0+cu130, Triton 3.7.1)
completed 520 tests; three noncurrent-device tests were skipped because it used
one GPU. All 132 input cases / 12,804 measurements completed, covering 36 weight
reduction configurations. Accuracy, fixed-call repeatability, training/inference
byte equality, and corresponding row-subset outputs/input gradients passed the
checks described above. These results do not establish full-model or ROCm parity.

For BF16 inputs [8192, 2688], isolated weight-reduction CUDA Graph medians were:

| Implementation | Time |
| --- | ---: |
| PyTorch `sum` | 0.04829 ms |
| Default SEQUENTIAL | 0.96434 ms |
| Default TILED | 0.56728 ms |
| TILED, rows=64 / cols=64 / warps=8 | 0.12707 ms |
| Default PARALLEL | 0.03664 ms |
| PARALLEL, rows=512 / cols=64 / warps=4 | 0.03264 ms |

The tuned parallel reduction was about 1.48x faster than PyTorch's sum. This is
an isolated device-timing comparison, not a public eager speedup. Separately,
BF16 [32768, 2688] public eager forward+backward measured 5.48097 ms for PyTorch
versus 1.06158 ms for that parallel configuration (5.16x), with close agreement
among four round medians. Raw reports remain external PR evidence.

The code explains a likely bottleneck: both original reductions launch only
`ceil(D / 128)` programs. TILED reduces serial loop iterations but still leaves
most of a large GPU unable to participate when D = 2688. PyTorch already uses
optimized compiled CUDA kernels: its [CUDA reduction implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Reduce.cuh)
can split the reduction across threads and multiple blocks, with vectorized
loads and staged partial results. The exact path used by this PyTorch build
has not been profiled; low program count is a code-based diagnosis, not a
hardware-counter measurement.

[Triton's LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
also separates partial weight gradients and final merging. Its lock-based
partial accumulation is not copied here: this experiment assigns fixed row
ranges and unique partial-buffer slots to avoid schedule-dependent additions.

The isolated sweep favors SEQUENTIAL at M=1, TILED at sampled M=16..128, and
PARALLEL at sampled M>=512 for D=2688. These are sampled points, not established
dispatch intervals. Eager backward/combined timings still show substantial
round-to-round variation even though identical-forward controls passed. The
subsequent boundary experiment completed 527 tests (three two-GPU skips) and
75 input cases / 6,600 measurements, covering all ten shortlisted configurations
through both eager and complete captured calls. All 2,475 public graph records
passed eager/captured byte equality and repeated-replay checks. Graph round
max/min stayed below 1.04; eager backward/combined timings still included
unstable measurements.

In the complete graph measurements, sampled M=1/2/4/8 favored SEQUENTIAL,
M=15..256 favored TILED, M=257 was close, and M>=384 favored PARALLEL. These
boundaries cannot be copied directly into eager dispatch: the extra launch and
allocation of PARALLEL can offset its GPU-time advantage in smaller eager calls.
The fixed PARALLEL configuration rows=512 / cols=64 / warps=4 measured public
eager forward+backward at [32768, 2688] in 1.07235 / 1.07321 / 1.30809 ms for
FP16 / BF16 / FP32 respectively, versus PyTorch's 5.39576 / 5.40082 / 4.50619 ms
(5.03x / 5.03x / 3.44x). Each of these measurements had round spread below 0.1%.
Raw reports remain external evidence.

The expanded dispatch experiment completed 538 tests (three two-GPU skips),
150 run-1 cases / 8400 measurements, and 51 reversed-order run-2 cases / 2856
measurements. The separate smoke added 56 measurements. Training/inference,
row invariance, reduction repeatability, and all public graph checks passed.
All 357 common public forward+backward graph records changed by less than 5%
between the two runs, while many small/medium eager records were unstable.
Graph winners therefore do not define the eager policy's crossover points.

For [65536, 2688], the mean of the two runs' eager forward+backward median times
with a **fixed explicit PARALLEL (512/64/4)** configuration was:

| Input | Eager PyTorch | Explicit PARALLEL | Speedup |
| --- | ---: | ---: | ---: |
| FP16 | 10.549 ms | 2.102 ms | 5.02x |
| BF16 | 10.553 ms | 2.100 ms | 5.03x |
| FP32 | 8.795 ms | 2.563 ms | 3.43x |

This is configuration evidence, not a performance measurement of the newly
connected automatic wrapper. Automatic boundary tests and a short public
benchmark remain the next GPU regression gate; a full sweep need not be repeated.

## Benchmark

For the short automatic-policy regression, compare only the public automatic Op
and eager PyTorch. This command has 21 cases / 126 measurements, with no isolated
reduction sweeps or CUDA Graph replay:

```bash
uv run --no-sync python benchmarks/benchmark_fused_add_rmsnorm.py \
  --auto-only --rows 8 9 32 33 16383 16384 16385 --cols 2688 \
  --rounds 4 --warmup 10 --repeat 50 \
  --output-dir reports/fused-rmsnorm-auto
```

`--auto-only` records the actual strategy and tile/warp settings per case. It
measures forward, backward, and forward+backward, including public dispatch and
allocation with warmed policy/device caches. Add `--public-graph` only for the
separate graph diagnostic (252 measurements for the same plan). A small fallback
check can use `--rows 33 16384 --cols 2689 --dtypes bf16`. Correctness tests cover
other widths, empty inputs, unknown devices, mixed dtypes and explicit overrides.
`--auto-only` is mutually exclusive with all configuration-sweep flags below.

The tuning experiments remain available for inspecting the selection evidence:

```bash
uv run --no-sync python benchmarks/benchmark_fused_add_rmsnorm.py --dry-run
uv run --no-sync python benchmarks/benchmark_fused_add_rmsnorm.py --preset smoke
uv run --no-sync python benchmarks/benchmark_fused_add_rmsnorm.py --preset model --sweep-configs
uv run --no-sync python benchmarks/benchmark_fused_add_rmsnorm.py --preset tuning --sweep-configs --dry-run
uv run --no-sync python benchmarks/benchmark_fused_add_rmsnorm.py \
  --preset boundary --shortlist-configs --public-graph --dry-run
uv run --no-sync python benchmarks/benchmark_fused_add_rmsnorm.py \
  --preset dispatch --finalist-configs --public-graph --dry-run
```

Use distinct `--output-dir` values for separate runs. Presets cover:

| Preset | M (rows) | D (features) |
| --- | --- | --- |
| smoke | 1, 33 | 129, 2688 |
| model (default) | 1, 32, 33, 128, 1024, 8192 | 129, 2688 |
| tuning | 1, 16, 31, 32, 33, 128, 512, 1024, 2048, 8192, 32768 | 128, 129, 2688, 4096 |
| boundary | 1, 2, 4, 8, 15, 16, 17, 31, 32, 33, 64, 128, 129, 192, 255, 256, 257, 384, 511, 512, 513, 1024, 2048, 8192, 32768 | 2688 |
| dispatch | 35 model-width row cases, plus 15 sparse other-width cases; see below | 2688; sparse 129, 2687, 2689, 4096, 8193 |

Every preset uses FP16/BF16/FP32 inputs, FP32 weights/upstreams, and eps = 1e-5.
Width 2688 comes from the model; other widths and 31/32/33 rows exercise tile
boundaries. `--rows`, `--cols`, and `--dtypes` can narrow or extend coverage.
The explicit `config.shapes` list in JSON is authoritative for sparse plans.

The dispatch plan fills the remaining eager crossover rather than repeating all
36 configurations. For D=2688 it keeps small-row controls and densely samples
M=2048..8192: 2048, 3072, 4096, 6144 and 8192 each include the immediately
preceding/following row, with 1536, 2560, 3584, 5120 and 7168 as additional
anchors. M=16384/32768/65536 check the large-row path beyond the previous maximum.
For each synthetic width 129/2687/2689/4096/8193 it tests M=1/2048/8192 to probe
column tails, model-width neighbors and widths beyond the previous sweep.
Those are fallback-policy evidence, not a claim that every width is tuned.
Empty inputs and invalid metadata remain correctness tests, not timing cases.

Dispatch uses 50 explicit shapes, not their full Cartesian product. Supplying
either `--rows` or `--cols` replaces this sparse plan with the specified grid
(unspecified columns default to 2688; unspecified rows use the 35 model rows).
`--reverse-cases` reverses the entire dtype/shape sequence without dropping cases.

Without a sweep, the benchmark compares PyTorch and the three default strategy
configurations. `--sweep-configs` tests 36 unique reduction configurations:

| Strategy | Row block | Column block | num_warps |
| --- | --- | --- | --- |
| SEQUENTIAL | unused | 32, 64, 128, 256 | 4, 8 |
| TILED | 16, 32, 64 | 64, 128 | 4, 8 |
| PARALLEL | 64, 128, 256, 512 | 64, 128 | 4, 8 |

`--shortlist-configs` is mutually exclusive with `--sweep-configs`. It keeps
the three defaults plus seven candidates from the H100 sweep: TILED
(rows=32/cols=64/warps=4, 64/64/4, 64/64/8) and PARALLEL
(rows=64/128/256/512, cols=64, warps=4). All ten configurations are measured
through the public operator; none is discarded based only on isolated timing.
The boundary preset with this shortlist and `--public-graph` has 75 input cases
and 6,600 measurements, versus 12,804 in the preceding broad sweep.

`--finalist-configs` is mutually exclusive with both other configuration flags.
It retains the three defaults as controls and adds TILED (32/64/4 and 64/64/8)
and PARALLEL (512/64/4). All six are measured through every public scope;
there is no per-case pruning based on isolated reduction speed. These fixed
finalists avoid turning sub-percent differences between large-row PARALLEL
configurations into a complicated policy. Dispatch plus finalists and
`--public-graph` produces 150 input cases / 8,400 measurements.

The completed selection experiment used eight rounds and then a fresh process
with reversed case order and a second seed for the key boundaries:

```bash
uv run --no-sync python benchmarks/benchmark_fused_add_rmsnorm.py \
  --preset dispatch --finalist-configs --public-graph \
  --rounds 8 --warmup 10 --repeat 50 --graph-unroll 16 \
  --output-dir reports/fused-rmsnorm-dispatch/run1
uv run --no-sync python benchmarks/benchmark_fused_add_rmsnorm.py \
  --preset dispatch --finalist-configs --public-graph \
  --rows 8 16 256 384 1536 2048 2049 2560 3072 3584 4096 5120 6144 7168 8192 8193 65536 \
  --rounds 8 --warmup 10 --repeat 50 --graph-unroll 16 --seed 435 --reverse-cases \
  --output-dir reports/fused-rmsnorm-dispatch/run2
```

Run 2 covers 51 cases / 2,856 measurements; both runs total 201 cases / 11,256
measurements. The plan targets the H100 80 GB used for previous tuning. Each
completed case prints its elapsed time and the cumulative duration, with counts
and accuracy gates preserved in the checkpointed JSON. If a run fails, its
partial report remains incomplete; do not infer success from file existence.

The measurement scopes are deliberately separate:

1. **Weight reduction:** `torch.sum` and each reduction configuration receive
   identical independently computed FP32 contributions and preallocated outputs.
   PARALLEL's partial workspace is also preallocated. Eager and CUDA Graph
   timings are both recorded, along with configuration, workspace size and
   launch program counts. Every candidate must pass an FP64-reference check,
   repeated-call byte equality, and a post-capture correctness check.
2. **Public eager operator:** PyTorch and all default strategies are compared
   for forward, backward and forward+backward. A sweep also takes the fastest
   reduction-graph candidate per strategy and measures it as `*_tuned` through
   the public API. These candidates must pass output/gradient accuracy,
   training/inference byte equality, and row-subset/permutation checks. Public
   timings include allocation and dispatch, including PARALLEL workspace
   allocation. `--shortlist-configs`/`--finalist-configs` measure every fixed candidate.
   Neither selection mechanism is a production dispatch map.
3. **Common forward diagnostic:** PyTorch and the common Triton forward are
   timed under CUDA Graph replay. All eager Triton forward timings are also
   compared as a control: a spread over 15% is flagged for rechecking before
   making strategy decisions. This flag is a heuristic, not a significance test.
4. **Complete public CUDA Graph diagnostic (`--public-graph`):** Every public
   candidate and PyTorch are captured for forward, backward and forward+backward.
   For backward-only capture, a prebuilt autograd graph is created on the capture
   stream, since backward inherits its forward's stream. Both gradient modes
   use fresh detached input leaves sharing the original immutable data, so live
   eager graphs cannot supply stale `AccumulateGrad` stream metadata. All warmup
   and eager comparison calls use the capture stream too. Combined capture records
   both directions. Replay excludes Python autograd traversal and allocation
   decisions, so these speedups are kept separate from eager speedups. Captured
   returns must match an independent eager call bitwise, remain identical on
   another replay, and pass reference accuracy checks again after timing. Only
   the last unrolled return is retained; inputs/prebuilt graphs and graph outputs
   stay alive throughout replay. The combined call returns gradients; outputs
   are checked separately by the forward captures.

Round max/min ratios are shown for every measurement. JSON lists all measurements
whose round medians differ by more than 15%, separately for eager and graph timing.
A passing identical-forward control alone no longer hides unstable backward
measurements. The threshold identifies cases to revisit; it is not a statistical
confidence interval and does not automatically select a production strategy.

The report also compares SEQUENTIAL/TILED and TILED/PARALLEL separately for
public backward/combined and eager/graph measurements. It compares the fastest
measured configuration within each family, keeps same-round latency ratios,
and labels the comparison `inconclusive` unless there are at least four rounds,
a median advantage of at least 5%, all matching rounds favor the same family,
both providers' round spreads are at most 15%, and instrumented wall time agrees
on direction. Wall time includes event instrumentation. This is a conservative
screening heuristic, not statistical significance or an automatic selector.
An independent reversed-order run must support the proposed boundaries; a
Graph candidate never substitutes for an inconclusive eager result.

Default timing uses 4 rounds, 10 warmups and 50 measured repetitions per provider
per round. Provider order reverses in pairs and rotates between pairs. Python
GC is disabled during measurement and restored afterwards. The reported median
is the median of round medians; raw samples, sample standard deviation, round
spread/order and wall time per instrumented call remain in JSON. There is no
`torch.compile` or explicit cache eviction. Compilation and correctness checks
are outside timing; small eager cases may include host dispatch gaps.

Graph diagnostics capture 16 calls per replay by default (`--graph-unroll`) and
divide the event time accordingly. Compare like timing modes and scopes only;
graph kernel timings are not public eager end-to-end speedups. This follows the
same host-overhead motivation as [Triton's CUDA Graph benchmark helper](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench_cudagraph.html).

In eager timing, forward uses `no_grad`; backward reuses a prebuilt graph;
forward+backward builds a fresh graph per call. Public extra peak allocation excludes inputs and prebuilt
graphs. For large row sums, reduction and weight-gradient absolute tolerance is
`2e-5 * sqrt(rows)`, with actual errors recorded. FP32 weight relative tolerance
is `2e-5`. This diagnostic tolerance does not replace the model's strict gate,
and repeatability within one configuration does not imply bitwise equality
between reduction configurations or separately reduced microbatches.

The default model run has 36 cases / 792 measurements. Model plus configuration
sweep has 3492 measurements; the full tuning sweep has 132 cases / 12804
measurements and is substantially longer. `--dry-run` prints the planned counts
without requiring a GPU. Reports are checkpointed after each completed case;
`complete` becomes true only after the whole plan finishes.

Reports default to `reports/fused-add-rmsnorm/{report.md,results.json}`. Attach
results to the PR instead of committing them. The complete-graph/boundary
experiment and expanded dispatch experiment passed on H100. The newly connected
automatic policy needs its targeted GPU correctness/performance regression.
Model cast-point alignment, ROCm and full-model/distributed validation also
remain separate work.
