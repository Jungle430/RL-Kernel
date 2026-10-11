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

```python
from rl_engine.runtime.registry import kernel_registry

op = kernel_registry.get_op("softcapped_selected_logprob", device=logits.device)
selected_logprob = op(logits, token_ids)
selected_logprob_cap20 = op(logits, token_ids, softcap=20.0)
```

Omitting `softcap` uses `30.0`. Both registry-created and directly constructed
Ops accept another cap at call time:

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
`ForwardConfigKey` records device identity, dtype, inclusive minimum/maximum row
counts and an **exact** vocabulary size. Lookup checks device-specific entries,
then `default`, then falls back to ROW. Unlisted widths or row counts outside
configured ranges remain supported; neither dimension is rounded or clamped.
Overlapping rules within a scope raise an error. Device-query failures propagate.

Selection is cached for 1024 exact metadata combinations. It does not depend on
softcap, gradient mode, input values or runtime timing. Caps reuse the same
launch schedules. GPU names are cached by backend
and device index. After changing the map or final fallback at runtime, call
`select_softcapped_logprob_strategy.cache_clear()`.

The default map is based on H100 80GB HBM3 measurements,
with `softcap=30.0`, PyTorch 2.13.0+cu130 and Triton 3.7.1: 837 eager cases and 27 CUDA Graph
checks. Range endpoints describe the measured evidence, not proven hardware
crossovers; intermediate row counts are interpolated. For Gemma's `V=262144`,
PARALLEL is selected for rows 1–4096 with FP16/BF16 and 1–2049 with FP32.
Other devices inherit this H100-derived default; model-specific overrides can
be added after measuring them. No benchmark reports are loaded at runtime.
A separate automatic-policy check completed 39 eager and 12 CUDA Graph cases.
All output/gradient accuracy and strategy bitwise checks passed; selection
matched every conclusive ROW/PARALLEL winner under the per-round 5% criterion.
This does not establish optimal performance for every shape or execution mode.
The existing measurements also do not establish performance for other caps;
use `--softcap` in the benchmark to measure those separately.

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

# Inspect the benchmark plan without a GPU.
uv run --no-sync python benchmarks/operators/logprob/benchmark_softcapped_selected_logprob.py --list-cases

# Small GPU smoke run, covering all three timing modes.
uv run --no-sync python benchmarks/operators/logprob/benchmark_softcapped_selected_logprob.py \
  --dtypes bf16 --shapes 3x1025 --warmup 1 --repeat 2 \
  --output-dir reports/softcapped-selected-logprob/smoke

# Default comparison; --shapes, --dtypes and --modes can narrow or extend it.
uv run --no-sync python benchmarks/operators/logprob/benchmark_softcapped_selected_logprob.py \
  --output-dir reports/softcapped-selected-logprob

# Compare another model's cap with the same PyTorch reference and timing modes.
uv run --no-sync python benchmarks/operators/logprob/benchmark_softcapped_selected_logprob.py \
  --softcap 50 --output-dir reports/softcapped-selected-logprob-cap50
```

Operator tests cover an independent FP64 reference, random upstream gradients,
the shared `logprob` tolerance contract, batch/layout invariance, saved-tensor
reuse, strategy boundaries and indexing guards. GPU tests skip on CPU hosts;
a GPU host that cannot load Triton fails rather than silently skipping.
Nondefault-cap cases cover reference outputs and gradients, all four strategies
and automatic selection, training/inference equality, batch-split strategy
changes, and two-GPU launches. The default reference is also compared bitwise
against the original fixed-30 expression. These GPU checks must pass on the
target hardware before extending the existing validation claims to new caps.

The benchmark compares eager `NativeSoftcappedSelectedLogprobOp` with the public
Triton automatic wrapper on the same CUDA/ROCm GPU. Defaults cover five shapes
(`1x1025`, `1x262144`, `16x262144`, `128x262144`, `1024x262144`), all three input
dtypes and forward/backward/forward+backward: 15 input cases and 45 comparisons.
Inputs are contiguous and reused without explicit cache eviction. No
`torch.compile`, graph capture or tuning sweep is part of this entry point.

It reuses `PerformanceProfiler` accelerator-event timing and checks output and
random-upstream gradient accuracy before timing, even with `--modes forward`.
Forward runs under `no_grad`; backward uses a prebuilt retained graph;
forward+backward constructs a fresh graph per call. Compilation and input
preparation are outside timing. Timed wrappers include allocation and dispatch
with a warm selector cache; small inputs can be affected by host launch gaps.

JSON/Markdown reports record the selected strategy, median latency, speedup,
extra peak allocation, softcap and the hardware/software environment. JSON also retains
sample standard deviations and accuracy errors. Extra allocation excludes
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
