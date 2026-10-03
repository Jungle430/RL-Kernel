# Softcapped selected logprob

Gemma's fixed-30 softcap followed by one selected-token log-probability per row
([issue #415](https://github.com/RL-Align/RL-Kernel/issues/415)):

$$
s_{ij}=30\tanh(x_{ij}/30),\qquad
\ell_i=s_{i,t_i}-\log\sum_j e^{s_{ij}}.
$$

With upstream gradient $g_i=\partial L/\partial\ell_i$ and
$p_{ij}=\exp(s_{ij}-\log\sum_k e^{s_{ik}})$:

$$
\frac{\partial L}{\partial x_{ij}}
=g_i\bigl(\mathbf{1}_{j=t_i}-p_{ij}\bigr)\bigl(1-\tanh^2(x_{ij}/30)\bigr).
$$

| Value | Shape | Dtype |
| --- | --- | --- |
| `logits` | `[M, V]`, `M >= 0`, `V > 0` | FP16 / BF16 / FP32 |
| `token_ids` | `[M]`, values in `[0, V)` | int64, same device as logits |
| Output | `[M]` | FP32 |
| Upstream gradient accepted by the backward launcher | `[M]` | FP16 / BF16 / FP32; computed in FP32 |
| Input gradient | `[M, V]` | Same dtype as logits |

Inputs are finite logits and valid indices; ignore-index/masking semantics are
not part of this interface. Noncontiguous inputs and upstream gradients are
accepted; Triton copies them to contiguous storage. Inputs are not modified.

```python
from rl_engine.kernels.registry import kernel_registry

op = kernel_registry.get_op("softcapped_selected_logprob", device=logits.device)
selected_logprob = op(logits, token_ids)
```

The registry follows `final_logit_softcap`: CUDA/ROCm prefer Triton, with native
PyTorch fallback if loading fails; CPU/MUSA/NPU use native PyTorch. The Triton
wrapper accepts the same device-type list as softcap. This list is not evidence
of hardware qualification. For explicit Triton validation, instantiate
`TritonSoftcappedSelectedLogprobOp` from `rl_engine.kernels.ops.triton.loss`
directly; the tests and benchmark do this, without timing a native fallback.

Default Triton forward uses one program per row, fixed 1024-element vocabulary tiles, four warps,
FP32 tile reductions and ascending FP32 accumulation, with FP fusion disabled.
Kernels specialize on the actual vocabulary width from `logits.shape[1]`
via `tl.constexpr` and retain masked boundary handling. Default forward uses an ordinary loop.
The fixed cap bounds scores to `[-30, 30]`, keeping the direct exponential sum
in range for the 262144-token vocabulary. Forward also saves one FP32
`log_sum_exp` per row (`4 * M` bytes of tensor data) for backward. Backward reuses
it with a `(M, ceil(V / 1024))` grid: each program computes one vocabulary tile
in one row and writes disjoint gradient elements, without a row loop, atomics,
or a merge kernel. Backward also uses four warps with FP fusion disabled and
casts gradients only on store. Autograd supports first-order gradients.
Native PyTorch supplies its own autograd and reduction schedule.

An experimental forward is available through an explicit constructor option;
the registry continues to use the row-loop default:

```python
from rl_engine.kernels.ops.triton.loss import TritonSoftcappedSelectedLogprobOp

row_op = TritonSoftcappedSelectedLogprobOp(forward_impl="row")
parallel_op = TritonSoftcappedSelectedLogprobOp(forward_impl="parallel")
```

Parallel forward launches a `(M, ceil(V / 1024))` grid to compute the same FP32
tile sums, then a second kernel accumulates them per row in ascending order,
starting from zero. It deliberately preserves the original addition order
rather than replacing the merge with a tree reduction or atomics. Scratch is
`4 * M * ceil(V / 1024)` bytes (1024 bytes per row for `V = 262144`) and is not
saved for backward. Both launches use the current stream and four warps with
FP fusion disabled. Both forward implementations save the same `[M]` FP32
`log_sum_exp` interface and share the tiled backward. No implementation is
selected automatically by row count, device occupancy or gradient mode.
GPU tests and the benchmark require bitwise agreement between the forward
variants on one device. Cross-platform equality requires separate validation.

```bash
python -m pytest tests/gemma/test_softcapped_selected_logprob*.py -q -rs
python scripts/check_operator.py --op softcapped_selected_logprob --candidate triton \
  --device cuda --dtype bf16 --batch 2 --seq 3 --vocab 262144 --check-grad
python benchmarks/benchmark_softcapped_selected_logprob.py \
  --shapes 1x1025 --dtypes fp32 --warmup 3 --repeat 5 \
  --output-dir ../softcapped-logprob-results/smoke
python benchmarks/benchmark_softcapped_selected_logprob.py \
  --output-dir ../softcapped-logprob-results/full
python benchmarks/benchmark_softcapped_selected_logprob.py \
  --modes forward --shapes 1x262144 4x262144 16x262144 64x262144 \
  --output-dir ../softcapped-logprob-results/forward-comparison
python benchmarks/benchmark_softcapped_selected_logprob.py \
  --suite dispatch --list-cases
python benchmarks/benchmark_softcapped_selected_logprob.py \
  --suite dispatch --output-dir ../softcapped-logprob-results/dispatch
python benchmarks/benchmark_softcapped_selected_logprob.py \
  --suite dispatch --rows 1 4 16 64 256 --vocab-sizes 4096 6144 8192 12288 16384 \
  --rounds 8 --warmup 50 --repeat 200 \
  --output-dir ../softcapped-logprob-results/refine
```

Tests include an independent FP64 reference, random upstream gradients, the
shared `logprob` tolerance contract, and exact per-row batch/layout invariance.
These are operator-level checks, not full-model training/rollout or TP acceptance.
GPU tests skip on CPU hosts and fail if a GPU host cannot load the Triton backend.

The benchmark compares eager PyTorch, the existing Triton softcap + logprob
sequence, fused row-loop forward and fused parallel forward, for forward,
backward and both together. Use `--modes forward` for a forward-only timing run;
output and gradient correctness are still checked first. The parallel timed
call includes scratch allocation, both kernel launches and the merge; no
scratch is preallocated outside it. The event timer includes accelerator work
and possible host dispatch gaps, not a separate synchronized CPU wall-time metric.
The benchmark checks native-relative accuracy and bitwise equality of the two
fused variants' output, saved statistics and gradient before timing. It reports
median latency, sample standard deviation, incremental peak allocation and
speedups. With multiple rounds, the table uses the median of round medians,
maximum within-round sample standard deviation and maximum extra peak allocation;
every round's measurements remain in JSON. `Row/parallel` is row-loop latency divided by parallel latency, so
values below one expose a regression. JSON and Markdown reports
include environment/source fingerprints. Keep generated reports outside the
checkout and attach measured evidence to the PR after GPU validation.

The `dispatch` suite measures forward by default to locate where vocabulary
parallelism pays off. It still checks output and backward accuracy, plus bitwise
equality of output, saved statistics and gradients, before timing every case.
All four implementations are measured, including both baselines.

| Axis | Dispatch cases | Purpose |
| --- | --- | --- |
| Input dtype | FP16, BF16, FP32 | Compare input bandwidth/cast costs; all math and outputs remain FP32 |
| Rows `M` | 1, 4, 16, 64, 256 | Check whether existing parallelism across rows changes the winner |
| Small widths and tails `V` | 512, 1023, 1024, 1025, 2048, 4096, 4097 | Expose launch overhead and tile-boundary effects |
| Intermediate widths `V` | 8192, 16384, 32768, 32769, 65536 | Locate the crossover region |
| Large widths `V` | 128256, 131072, 151936, 256000, 262144 | Cover large vocabularies, including Gemma's target width |

This is 255 dtype/shape cases, each measured in four rounds with 25 warmups and
100 repetitions per implementation per round. Implementation order rotates so
each occupies every position once per four rounds. Use `--list-cases` to inspect
the plan without a GPU. `--rows` and `--vocab-sizes` replace the sweep axes;
`--shapes` supplies explicit cases instead. `--dtypes` can narrow the sweep.
Add `--modes forward forward_backward` to also verify end-to-end training costs.
The `standard` suite retains the original five shapes and all three modes.

The report's forward selection table groups results by dtype and vocabulary
width, retaining each row count. A per-shape candidate must win by at least
1.05x in every round, with within-round std/median and the range of round medians
divided by their median both at most 10%. Complete four-round rotations and at
least two repetitions are required. This conservative screen is not a statistical
confidence interval. Close, noisy or incomplete results stay `inconclusive`;
opposite winners at different row counts are labeled `depends_on_rows`.
The JSON contains the screening criteria and individual reasons.

These candidates apply only to measured shapes on that GPU/software and
contiguous inputs. They do not install an automatic selector or infer a monotonic
vocabulary threshold. Refine around observed transitions and repeat measurements
before selecting a rule; check whether a rule based on only dtype and vocabulary
can cover all tested row counts. Keep hardware results separate, and validate
bitwise invariance again when introducing any selector. Backward uses the same
kernel for both forward variants and cannot establish a forward dispatch rule.
Completed cases are checkpointed outside timing; interrupted reports carry
`complete: false` and list missing row counts in already-started groups.
