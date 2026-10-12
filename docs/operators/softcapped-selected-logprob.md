# Softcapped selected logprob

A shared logprob operator that applies a configurable softcap and returns one
selected-token log-probability per row. It was introduced for Gemma
([issue #415](https://github.com/RL-Align/RL-Kernel/issues/415)); the interface
has no model-topology dependency. The scalar `softcap` is denoted by $c$ below
and defaults to `30.0`:

$$
s_{ij}=c\tanh(x_{ij}/c),\qquad
\ell_i=s_{i,t_i}-\log\sum_j e^{s_{ij}}.
$$

With upstream gradient $g_i=\partial L/\partial\ell_i$ and
$p_{ij}=\exp(s_{ij}-\log\sum_k e^{s_{ik}})$:

$$
\frac{\partial L}{\partial x_{ij}}
=g_i\bigl(\mathbf{1}_{j=t_i}-p_{ij}\bigr)\bigl(1-\tanh^2(x_{ij}/c)\bigr).
$$

## Interface and precision

| Value | Shape | Dtype |
| --- | --- | --- |
| `logits` | `[M, V]`, `M >= 0`, `V > 0` | FP16 / BF16 / FP32 |
| `token_ids` | `[M]`, values in `[0, V)` | int64, same device as logits |
| `softcap` forward keyword argument | Scalar; default `30.0` | Positive real number; no gradient |
| Output | `[M]` | FP32 |
| Upstream gradient accepted by the backward launcher | `[M]` | FP16 / BF16 / FP32; computed in FP32 |
| Input gradient | `[M, V]` | Same dtype as logits |

Inputs must contain finite logits and valid indices. There is no ignore-index
or token-masking argument. Noncontiguous inputs and upstream gradients are
accepted; Triton copies them to contiguous storage. Inputs are not modified.
Autograd supports first-order gradients. Native PyTorch supplies its own autograd.

`softcap` is passed per call, independently of the strategy/tuning configuration.
The same Op can serve calls with different caps. Each autograd invocation keeps
its own cap for backward; the Op does not store or mutate this value.
It must be finite and at least `torch.finfo(torch.float32).tiny`. Both
implementations preserve the original direct `exp -> sum -> log` calculation.
To avoid overflow or flushing every exponential in a negative saturated row to
zero, they reject cap/vocabulary combinations unless
`exp(-c) >= 2 * float32.tiny` and `V * exp(c) <= float32.max / 2`.
The bounds are checked in log space, using only the scalar and shape metadata.
For example, both `30` and `50` are accepted at `V=262144`; `100` is not.
Supporting caps outside this range would require a stable, shifted reduction.

The PyTorch numerical reference creates an FP32 `softcap` tensor on the logits
device and uses `torch.div(logits.float(), divisor)`. This avoids the CUDA
Python-scalar division path, which multiplies by a precomputed rounded reciprocal
([PyTorch implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu)).
The reference remains entirely native PyTorch with FP32 intermediates and native
autograd. Triton retains `tl.div_rn`; it does not adopt reciprocal multiplication
to reproduce the old reference's rounding. WS1 accuracy compares with this FP32
reference. Independent FP64 mathematical tests have separate, explicitly stated
targets; they do not redefine the shared contract's reference or tolerances.
Native reductions can still differ with batching, so reference accuracy and
Triton's bitwise training/inference and batch invariance are separate checks.

```python
from rl_engine.runtime.registry import kernel_registry

op = kernel_registry.get_op("softcapped_selected_logprob", device=logits.device)
selected_logprob = op(logits, token_ids)
selected_logprob_cap20 = op(logits, token_ids, softcap=20.0)
```

Omitting `softcap` uses `30.0`. Both registry-created and directly constructed
Ops accept another cap at call time. The native implementation function requires
an explicit `softcap`; its default is supplied only by the Op's `forward` method:

```python
from rl_engine.reference.logprob.softcapped_selected_logprob import (
    NativeSoftcappedSelectedLogprobOp,
)
from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
    TritonSoftcappedSelectedLogprobOp,
)

reference = NativeSoftcappedSelectedLogprobOp()
op = TritonSoftcappedSelectedLogprobOp()
expected = reference(logits, token_ids, softcap=50.0)
selected_logprob = op(logits, token_ids, softcap=50.0)
```

The registry prefers Triton on CUDA/ROCm, falling back to native PyTorch if the
backend cannot load. CPU/MUSA/NPU use native PyTorch. The Triton wrapper accepts
`cuda`, `hip`, `xpu` and `musa`, following `final_logit_softcap`; acceptance does
not establish validation on those devices. Performance tuning targets H100;
ROCm qualification remains separate. Tests and benchmarks instantiate Triton
directly so they cannot silently measure a native fallback.

## Triton implementation

All softcap, exponential, reduction and backward arithmetic uses FP32. Vocabulary
tiles contain 1024 elements and use four warps, with FP fusion disabled. The
vocabulary width and softcap are compile-time constants; partial tiles are masked.
Different caps may compile separate kernels. The default `30.0` retains the
original arithmetic and tile-reduction order.

- `ROW`: one program processes each row, reducing each tile and accumulating
  its sum in ascending vocabulary order.
- `PARALLEL`: separate programs compute row/tile sums. A second kernel merges
  them in the same order as ROW. Temporary storage is
  `4 * M * ceil(V / 1024)` bytes and is released after forward.

Both paths save one FP32 `log_sum_exp` per row (`4 * M` bytes) and use the same
backward kernel. Backward processes disjoint row/vocabulary tiles, reusing the
saved statistics without repeating the reduction. It casts gradients only on
store. Neither path materializes a full softcapped or probability tensor.
The cap bounds scores to `[-c, c]`; host validation checks the FP32 range above.
Autograd saves the cap used by each forward call, so later calls with a different
cap do not change that graph's backward. Direct backward-launcher callers must
pass the same cap used to compute the saved `log_sum_exp`.

ROW and PARALLEL preserve the same arithmetic order so changing the strategy
with batch size does not change the bits on the same tested device. Tests check
outputs, saved statistics and gradients, as well as forward equality with
and without gradient recording. This is operator-level coverage, not full-model
training/rollout, tensor-parallel or cross-device equivalence.

## Automatic strategy selection

```python
from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
    SoftcappedLogprobStrategy,
    TritonSoftcappedSelectedLogprobOp,
)

auto_op = TritonSoftcappedSelectedLogprobOp()  # forward_impl=None
row_op = TritonSoftcappedSelectedLogprobOp(forward_impl=SoftcappedLogprobStrategy.ROW)
parallel_op = TritonSoftcappedSelectedLogprobOp(forward_impl=SoftcappedLogprobStrategy.PARALLEL)
```

The static map and selector live alongside the shared Triton implementation in
`rl_engine/backends/shared/triton/logprob/softcapped_selected_logprob.py`.
`ForwardConfigKey` records device identity, dtype, inclusive row-count bounds and
inclusive **vocabulary tile-count bounds**. The selector computes
`n_tiles = (V + 1023) // 1024`; `max_tiles=None` means no upper bound. This describes
how many tiles the existing kernels process. It does **not** round the actual
vocabulary, pad the input, or change the arithmetic and reduction order.

Its `device` field uses two immutable variants, following the shared RMSNorm
configuration design. `SoftcappedLogprobDefaultDevice()` carries no metadata
and denotes the shared default. A concrete device is written as
`SoftcappedLogprobDevice(backend="cuda", gpu_name="NVIDIA H100 80GB HBM3")`.
It matches a backend and hardware model, not a device ordinal such as `cuda:0`.
ROCm uses `backend="rocm"`, even though PyTorch exposes it through `torch.cuda`.
There is no string sentinel or positional device tuple in the configuration.

All rule constructors name their fields explicitly:

```python
import torch

from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
    ForwardConfigKey,
    SoftcappedLogprobDefaultDevice,
)

key = ForwardConfigKey(
    device=SoftcappedLogprobDefaultDevice(),
    dtype=torch.bfloat16,
    min_rows=1,
    max_rows=4096,
    min_tiles=32,
    max_tiles=None,
)
```

The generalized candidate has three entries, one for each supported dtype:

| Rows M | Vocabulary tiles | FP16 / BF16 / FP32 |
| --- | --- | --- |
| 1–4096 | 32 or more | PARALLEL |
| All remaining supported shapes, including M=0 | Any | ROW fallback |

32 tiles begin at `V=31745`: `V=31744` fills exactly 31 tiles. Gemma's
`V=262144` uses 256 tiles; `V=202048` uses 198 tiles and shares the same policy.
A new model does not need an exact vocabulary entry. Larger unseen vocabularies
also use PARALLEL within the configured row range; batches above 4096 use ROW.
These are launch-policy bounds, not input-size restrictions or proven optimal
hardware crossovers.

Lookup checks device-specific entries, then the default-device variant, then
`DEFAULT_FORWARD_STRATEGY` (ROW). Device rules can specify a finite upper tile
bound and override a default PARALLEL choice with ROW. Rules at the same scope
and dtype must not overlap in **both** row and tile ranges. Ambiguous matches
raise an error; device-query failures propagate. Other devices inherit the
H100-derived default until device-specific measurements justify overrides.

Selection is cached for 1024 exact metadata combinations. It does not depend on
softcap, gradient mode, input values or runtime timing. GPU names are cached by
backend and device index. After changing the map or final fallback at runtime,
call `select_softcapped_logprob_strategy.cache_clear()`. No benchmark reports
are loaded at runtime.

The rule was checked offline against existing H100 80GB HBM3 measurements at
`softcap=30.0`, PyTorch 2.13.0+cu130 and Triton 3.7.1. Replaying the choices over
837 eager and 27 CUDA Graph cases agrees with every conclusive ROW/PARALLEL
winner in those reports. Based on their recorded medians, the chosen explicit
implementation is at most 2.73% slower than the faster of ROW/PARALLEL in the
eager sweep. A separate historical policy-check dataset (39 eager / 12 graph
cases) gives maxima of 0% / 1.89%. Near ties can now choose PARALLEL instead of
retaining an exact-width exception.

This is an **offline policy check**, not a new GPU execution or a timing of the
new AUTO wrapper. Earlier correctness results apply to the tested kernels and
shapes. Newly interpolated widths, the 31/32-tile boundary, large-vocabulary
extrapolation and nondefault caps still require GPU regression and performance
checks. The compact comparison below covers those gaps before treating this
candidate policy as validated. Performance on other hardware is unverified.

Two explicit experimental strategies remain available: `ROW_PIPELINED` requests
three loop stages with ROW arithmetic; `ROW_ACCUMULATE` accumulates a vector
before reducing once, changing rounding order. Neither is selected by the
automatic map. In particular, mixing ROW_ACCUMULATE into a batch-dependent map
would require resolving its different numerical results.

## Tests and benchmark

```bash
uv sync --extra dev
uv run --no-sync python -m pytest \
  tests/ops/logprob/test_softcapped_selected_logprob.py \
  tests/backends/shared/triton/logprob/test_softcapped_selected_logprob_config.py \
  tests/runtime/test_softcapped_selected_logprob_registry.py \
  tests/validation/operators/test_softcapped_selected_logprob_harness.py \
  tests/benchmarks/test_softcapped_selected_logprob_benchmark.py -q -rs
uv run --no-sync python tools/validation/operators/check_operator.py --op softcapped_selected_logprob \
  --candidate triton --device cuda --dtype bf16 --batch 2 --seq 16 --vocab 262144

# Additional mathematical checks, reported separately from FP32 acceptance.
# The known cap=70 discrepancy remains an ordinary failing assertion here.
uv run --no-sync python -m pytest \
  tests/ops/logprob/test_softcapped_selected_logprob_math.py -q -ra

# Inspect the benchmark plan without a GPU.
uv run --no-sync python benchmarks/operators/logprob/benchmark_softcapped_selected_logprob.py --list-cases

# Small GPU smoke run, covering all three timing modes.
uv run --no-sync python benchmarks/operators/logprob/benchmark_softcapped_selected_logprob.py \
  --dtypes bf16 --shapes 3x1025 --warmup 1 --repeat 2 \
  --output-dir reports/softcapped-selected-logprob/smoke

# Default comparison; --shapes, --dtypes and --modes can narrow or extend it.
uv run --no-sync python benchmarks/operators/logprob/benchmark_softcapped_selected_logprob.py \
  --output-dir reports/softcapped-selected-logprob

# Optional dispatch check: AUTO vs explicit ROW/PARALLEL and eager PyTorch.
# Three dtypes; row/tile boundaries, unseen widths and large-V extrapolation.
uv run --no-sync python benchmarks/operators/logprob/benchmark_softcapped_selected_logprob.py \
  --compare-strategies --modes forward --warmup 50 --repeat 200 \
  --shapes 16x31744 16x31745 16x32768 4095x32768 4096x32768 4097x32768 \
    16x65537 1024x202048 4096x202048 4097x202048 16x2097153 \
    4096x262144 8192x65536 \
  --output-dir reports/softcapped-selected-logprob-ranges

# Check a different cap at an unseen vocabulary width on both sides of M=4096.
uv run --no-sync python benchmarks/operators/logprob/benchmark_softcapped_selected_logprob.py \
  --softcap 20 --compare-strategies --modes forward --warmup 50 --repeat 200 \
  --shapes 16x202048 1024x202048 4096x202048 4097x202048 \
  --output-dir reports/softcapped-selected-logprob-cap20
```

Validation has three layers:

| Layer | Comparison and requirement | Location |
| --- | --- | --- |
| FP32 reference accuracy | Compare outputs and gradients with `NativeSoftcappedSelectedLogprobOp` on the same device. Promote the already-quantized inputs and upstreams to FP32, and keep reference gradients in FP32. Use the unchanged WS1 `logprob` tolerances. | `tests/ops/logprob/test_softcapped_selected_logprob.py` and the shared operator harness |
| Bitwise invariance | Compare training, `no_grad`, `inference_mode`, repeated backward, batch/chunk/layout changes and supported AUTO strategy switches. Require identical bytes within each dtype. | `tests/ops/logprob/test_softcapped_selected_logprob.py` |
| Independent FP64 mathematics | Evaluate stable FP64 `log_softmax` and its autograd gradients on the same quantized inputs, without calling the Native or Triton implementation. Check both Native and the public Triton strategies. | `tests/ops/logprob/test_softcapped_selected_logprob_math.py` |

The direct FP32 tests retain the existing tolerance selection: forward FP32
outputs use `atol=1e-5, rtol=0`; gradients use their output dtype's contract row
(FP32 `1e-5`, FP16 `5e-3`, BF16 `5e-2`, all with `rtol=0`). The FP32 reference
is not rounded to FP16/BF16 before comparison. The shared harness retains its
own contract selection by case dtype. Neither path permits private tolerance
relaxation. See the [FP32 reference definition](../contracts/ws1-numerical-precision-standard.md#3-definition-of-the-fp32-reference).

The extra FP64 tests preserve the earlier numerical targets: forward absolute
error at most `1e-5`; gradient absolute error at most `1e-5`, `5e-3` or `5e-2`
for FP32, FP16 or BF16 output respectively, with zero relative tolerance.
These are test-specific mathematical goals, not a project-wide FP64 contract.
They retain FP64 oracle values during comparison and do not borrow the WS1
resolver. Passing FP32 agreement does not automatically pass these stricter
checks. Running the whole test directory includes both files; the extra checks
are not skipped or marked `xfail` to obtain a green result.

Random upstream gradients, saved-tensor reuse, strategy boundaries and indexing
guards remain covered. GPU tests skip on CPU hosts; a GPU host that cannot load
Triton fails rather than silently skipping.
Nondefault-cap cases cover reference outputs and gradients, all four strategies
and automatic selection, training/inference equality, batch-split strategy
changes, and two-GPU launches. The configurable-cap consistency sweep uses
`1e-6, 1e-3, 0.125, 0.5, 1, 3.7, 10, 17.3, 20, 30, 39.5, 50, 64, 70`;
it requires bitwise training/`no_grad`/`inference_mode` agreement and repeated
backward agreement. Triton also compares individual rows against the batch,
and AUTO is checked while crossing a ROW/PARALLEL boundary with each cap.
Endpoint tests use `float32.tiny` and a cap `1e-4` below the allowed maximum
at `V=1, 1025, 202048, 262144`, covering saturation and rejecting values outside
the bounds. Thus the largest tested valid caps vary with V (about 75.6–86.6).
Native GPU reference checks cover the same 14-cap sweep at `V=1025, 202048`.
All these checks cover FP16/BF16/FP32 without relaxing the existing tolerances.
Two-GPU launch coverage retains caps `30` and `50`.
Division-stage tests compare with an independently
rounded FP64 quotient, including `39.5 / 50`. A deterministic wide-row regression
at `softcap=50`, `V=202048` checks both the revised FP32 reference and FP64.
Omitted and explicit `softcap=30` calls must agree bitwise. These GPU checks must
pass on the target hardware before extending validation claims to new caps.

The A6000 expanded sweep exposed an extra FP64-target failure at `softcap=70`,
`[M, V]=[17, 32767]`, seed `438`: its error against the FP64 oracle is about
`1.18e-5`, above the unchanged extra `1e-5` limit. Native also shows the discrepancy.
The ROW/PARALLEL batch-split bitwise checks passed, and the observed Triton/Native
forward difference stayed below the FP32 contract limit. The original AUTO
test now checks FP32 agreement; `test_fp64_math_cap70_rounding_regression`
retains the independent failure with separate output/gradient cases so an
output failure cannot prevent the gradient check. This reorganization does
not fix FP64 accuracy or establish precision qualification for every accepted
cap. Capture its intermediate values with:

```bash
python -m tools.validation.operators.diagnose_softcapped_precision \
  --output-dir reports/softcapped-precision
```

This diagnostic records both the native FP32 and Triton paths against FP64.
The instrumented ROW output and saved statistic must match production bitwise;
otherwise the trace fails and cannot attribute the production error. Its exit
status checks instrumentation, not accuracy: run the FP32 and FP64 pytest
layers separately to obtain their respective accuracy results.

`NativeSoftcappedSelectedLogprobOp` is both the numerical reference and the timed
PyTorch baseline. The benchmark calls the public Native and automatic Triton
Ops on the same CUDA/ROCm GPU, without maintaining another PyTorch implementation
or importing private input-validation helpers. Triton must pass output and
gradient checks against Native FP32 before timing. The accuracy check retains
FP32 reference gradients, while timing still calls both Ops with the original
FP16/BF16/FP32 inputs. Native's FP32 device-divisor
creation and transfer are included in forward and forward+backward timing;
backward timing reuses the divisor saved in the prebuilt graph. Reports identify
this division path; earlier scalar-division measurements used a different baseline.
Defaults cover five shapes
(`1x1025`, `1x262144`, `16x262144`, `128x262144`, `1024x262144`), all three input
dtypes and forward/backward/forward+backward: 15 input cases and 45 comparisons.
Inputs are contiguous and reused without explicit cache eviction. No
`torch.compile` or graph capture is used. The default run performs no strategy sweep.

It reuses `PerformanceProfiler` accelerator-event timing and checks output and
random-upstream gradient accuracy before timing, even with `--modes forward`.
Forward runs under `no_grad`; backward uses a prebuilt retained graph;
forward+backward constructs a fresh graph per call. Compilation and input
preparation are outside timing. Timed wrappers include allocation and dispatch
with a warm selector cache; small inputs can be affected by host launch gaps.

`--compare-strategies` optionally adds explicit ROW and PARALLEL to **forward**
timing. All four providers rotate through each timing position across four
rounds. The report gives the median of round medians, preserves individual
rounds and reference-accuracy checks, and displays AUTO latency divided by the
faster explicit strategy. Values above 1 expose dispatch overhead or a slower
choice. The aggregate `std_ms` is the largest within-round sample deviation,
not a confidence interval. Repeat near ties on the target GPU; the benchmark
does not modify the map. Backward and forward+backward retain the standard
PyTorch-vs-AUTO comparison even when this flag is enabled.

JSON/Markdown reports record the selected strategy, median latency, speedup,
extra peak allocation, softcap and the hardware/software environment. JSON also retains
sample standard deviations and `accuracy` errors against the native baseline.
Extra allocation excludes
inputs and prebuilt graphs. Cases are saved outside timing as they finish;
interrupted reports remain marked `complete: false`. Speedup is native latency
divided by Triton latency, so values below 1 expose a slowdown. When run from the
repository root, reports default to the directory
`reports/softcapped-selected-logprob/`. Attach performance evidence to the PR
rather than committing generated reports.

### Reports and validation evidence

See [PR #463](https://github.com/RL-Align/RL-Kernel/pull/463) for H100
training/inference consistency results, strategy-tuning evidence and performance
comparisons with eager PyTorch. The PR links the Markdown reports and JSON
measurements, and records the tested source commit and hardware/software
environment. ROCm validation remains pending.
