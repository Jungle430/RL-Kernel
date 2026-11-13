# Shared Index Select

`index_select` copies slices along an explicitly selected dimension. It serves
the H3 output-selection task and BAGEL's packed-token selections through the
same API described in [#680](https://github.com/RL-Align/RL-Kernel/issues/680).

## API and model boundaries

```python
from rl_engine.runtime.registry import kernel_registry

select = kernel_registry.get_op("index_select", device=x.device)
y = select(x, index, dim=dim)
```

Both `NativeIndexSelectOp.forward(x, index, *, dim)` and
`TritonIndexSelectOp.forward(x, index, *, dim)` require an explicit keyword
axis. Their `__call__` methods have the same signature.

| Use case | Input | Call | Output |
| --- | --- | --- | --- |
| H3 video/audio head output | `[B, S, D]` | `select(x, index, dim=1)` | `[B, K, D]` |
| BAGEL packed hidden states | `[S, D]` | `select(x, index, dim=0)` | `[K, D]` |
| BAGEL packed query states | `[S, H, D]` | `select(x, index, dim=0)` | `[K, H, D]` |
| Last-axis selection | `[..., D]` | `select(x, index, dim=-1)` | `[..., K]` |

In the pinned [H3 forward](https://github.com/huggingface/diffusers/blob/f53d552036a0d1bd5570782a39cd40cfabf112bc/src/diffusers/models/transformers/transformer_minimax_h3.py#L622-L627),
both output heads project every mixed-sequence row before selection. Their
FP32 output widths are 96 and 32. The model's RFC #420 work-item name and frozen
architecture fingerprint remain `packed_index_select`; the reusable operator
and registry key are now `index_select`.

[BAGEL](https://github.com/ByteDance-Seed/Bagel/blob/a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f/modeling/bagel/qwen2_navit.py#L419-L448)
selects packed understanding/generation tokens before branch-specific projection
or normalization. It also selects query slices with head dimensions. This
operator covers the selection itself; subsequent projection, normalization,
scatter-back and loss computation retain their own contracts.

The example `examples/models/minimax_h3/packed_index_select.py` calls
the shared operator with `dim=1`. This change supplies operator and registry
entries, not model rewrites. In particular the H3 full-model provider still uses
its tensor-method output selection; neither H3 nor BAGEL full-model integration
is qualified by these operator tests.

## Tensor and device contract

| Value | Requirements |
| --- | --- |
| `x` | Rank >= 1; FP32, FP16 or BF16; strided and expanded views accepted |
| `index` | Same-device 1-D int32/int64 `[K]`; entries in `[0, x.shape[dim])` |
| `dim` | Required keyword; negative axes count from the end of x |
| Output | x's shape with the selected axis replaced by K; contiguous; x's dtype |
| Upstream gradient | Output shape and dtype; strided and expanded views accepted |
| Input gradient | x's shape and dtype; contiguous; unselected coordinates are +0 |

The index vector is shared across all other coordinates. Duplicates are legal
and retain multiplicity. Negative indices do not wrap; indices must not be
modified between forward and backward.

`K=0` returns an empty output and zero input gradient. A zero-length selected
axis requires an empty index; empty non-selected axes also produce empty
tensors. Index bounds are checked even when another dimension is empty.
Metadata errors fail synchronously. CUDA bounds checks use asynchronous device
assertions, which can invalidate the CUDA context; invalid-index tests use
isolated subprocesses.

The Triton backend retains the original NVIDIA CUDA SM80+ eligibility.
Model generalization does not qualify ROCm or other hardware. Production
dispatch fails closed on unsupported platforms. On H100 NVL, a compiled exact-math
CUDA backend is preferred for eager calls in its bounded native subset. CUDA
Graph-captured forward calls and other eligible CUDA inputs use the shared Triton backend. A missing native extension preserves that fallback.
The PyTorch reference can be called explicitly, including on CPU:

```python
from rl_engine.reference.embedding.index_select import NativeIndexSelectOp

y = NativeIndexSelectOp()(x, index, dim=0)
```

Only first-order autograd is supported. Distributed selection and compilation/
export are not qualified. Fixed-input CUDA Graph forward, backward, and combined
replay are exercised by the operator benchmark; this does not qualify arbitrary
application graph integration.

## Computation and fixed accumulation order

For indexing purposes, flatten the dimensions before and after `dim`:

```text
x: [outer, axis, inner]       index: [K]
                y[outer, j, inner] = x[outer, index[j], inner]
y: [outer, K, inner]
```

This is a logical view: the kernel decodes coordinates using original strides,
without copying x into this layout. Contiguous inputs use a 2-D grid over outer
coordinates and flattened output tiles, so narrow selected slices can share a
coalesced write. Strided inputs retain a 2-D grid over blocks of `outer * K`
rows and inner tiles. Both paths cap the second grid axis and loop over remaining
tiles. These choices change ownership only, not copied values.
Forward only loads and stores, preserving bits without floating-point arithmetic.

For each source coordinate, let `j0 < j1 < ...` be the output positions selecting
it. Backward performs:

```text
a = FP32(+0)
a = round_fp32(a + FP32(grad[outer, j0, inner]))
a = round_fp32(a + FP32(grad[outer, j1, inner]))
...
dx[outer, source, inner] = cast_to_input_dtype(a)
```

Backward chooses grouping from shape metadata; it never reads index values on
CPU or caches a mutable index. All paths implement the same addition chain:

- **Small selection:** when K, axis size and outer count are bounded, each source
  slice finds its occurrences directly using integer minima. This avoids both a
  sort and a separate zero-fill launch. For the bounded scan, K is at most 1024
  and initial source/index comparisons are capped; large duplicate groups sort
  their integer positions locally before prefetching gradients.
- **Dense selection:** one integer workspace holds counts, bounded position
  slots, overflow-list heads, a work queue and links. Atomic ranks assign unique
  slots; they do not define addition order. The fifth occurrence queues a source
  exactly once. Short groups sort up to four integer positions with compare/
  exchanges; groups of 5–128 occurrences sort collected positions locally.
  Larger groups scan the original index in bounded tiles while carrying the
  same FP32 accumulator across tiles. Gradient loads are prefetched in batches
  of eight, then added individually in original occurrence order. Short and
  long groups have disjoint output ownership, allowing one combined launch.
  Large metadata domains use separate short/long specializations to avoid
  imposing the heavy path's register usage on all source rows.
- **Sparse or large integer domain:** retain stable sorting, so metadata storage
  depends on K rather than the full axis. Small vectors use packed integer keys
  with an explicit overflow bound; larger vectors use `torch.sort(stable=True)`.
  Only segment heads write, with the same sequential FP32 additions.

Dense metadata is used only when the axis is at most eight times K, signed
integer addressing/counts remain in range, and the workspace fits 128 MiB.
The launcher selects 128 position slots per source, or four slots when necessary
to respect that limit. Larger domains use stable sorting. Slot count, the
16/128-position local sorting networks and execution tiles affect metadata
processing, not floating-point association. Tests force both slot configurations
and the stable-sort fallback on identical data and compare gradient bits.

The dense path uses **integer atomics only**. Queue order and linked-list order
can vary; gradient operands are explicitly reordered before addition. Each
gradient coordinate has one writer in each active stage. There are no
floating-point atomics or sums across unrelated coordinates. FP16/BF16
gradients round once at the final store. Kernels run on the input device's current
stream, restoring the caller's current device.

The order is invariant to training/inference forward, repeat calls, batching
independent coordinates, checkpoint replay, and consistent source relabeling.
If `dim` itself is the batch axis, selecting duplicate batches reduces over that
axis: splitting those duplicate contributions between separate backward calls
is **not** covered by batch invariance. Reordering duplicate occurrences can
also change FP32 rounding. Fixed order is not exact real arithmetic.

`NativeIndexSelectOp.forward` uses PyTorch selection and explicitly ordered CPU
FP32 backward additions. `forward_fp32` supplies an independent accuracy oracle:
FP32 selection and CPU FP64 `index_add_`, cast to the input leaf's dtype.
Accuracy tests use FP32 leaves for this oracle, so low-precision gradients are
compared with an FP32 result. These references are not performance baselines.

## H100 NVL native backend

`H100IndexSelectOp` implements the same public contract through the precompiled
`rl_engine._C` extension. Build it with exact CUDA math:

```bash
RL_KERNEL_REQUIRE_EXT=1 python -m pip install --no-build-isolation -e .
```

The build omits this backend when fast math or FTZ is requested, including
relaxed options supplied through NVCC environment flags. A lightweight Python
autograd wrapper saves the index and shape; C++ owns validation, allocation,
device guards and kernel submission. Importing the backend does not compile it.

Native dispatch is limited to the tested `NVIDIA H100 NVL` device name and SM90,
rank at most 32, logical element counts below `2**31 - 1`, and bounded integer
workspace. Other CUDA devices or larger shapes retain the shared implementation.
The legacy registry selects from the actual input tensor's device at call time;
it does not cache the process's current GPU as the dispatch decision. Native
eager execution reduces host submission overhead. During CUDA Graph capture the
registry uses Triton for forward calls, whose combined device replay is faster
on most measured workloads. A backward-only capture retains the backend of its
earlier eager forward; capture does not replace an existing autograd node.
The strict native implementation also supports the tested Graph captures and is
available for explicit comparisons. The explicit semantic backend `cuda-h100-index-select` fails on unsupported hardware.

Small selections scan the original index with warp ballots. Larger selections
build bounded integer occurrence slots, sort those positions, then load gradients
in increasing original position. Large outputs separate short and queued long
groups to keep the common path's register use low. Contiguous forward copies and
short-group gradient reads/writes use aligned 16-byte storage packs; each element
still has its own FP32 accumulator. Overflow groups prefetch index windows and
scan the original vector instead of splitting the FP32 sum. Every coordinate has a single writer. Sequential
`__fadd_rn` operations preserve the FP32 rounding boundary, including subnormals;
the only cast to the input dtype occurs at the final store.

## Validation

```bash
export PYTHONPATH=.
python -m pytest -q tests/ops/embedding/test_index_select.py \
  tests/ops/embedding/test_index_select_h100_dispatch.py \
  tests/benchmarks/test_index_select_benchmark.py
python -m pytest -q tests/runtime/test_kernel_registry.py \
  tests/validation/operators/test_operator_inputs.py -k index_select
python tools/validation/operators/check_operator.py \
  --op index_select --candidate triton --device cuda --dtype bf16 \
  --select-shape 129 4 128 --select-dim 0 --check-grad --grad-mode random
# Explicit native backend on H100 NVL after building the extension
python tools/validation/operators/check_operator.py \
  --op index_select --candidate cuda --device cuda --dtype bf16 \
  --select-shape 129 4 128 --select-dim 0 --check-grad --grad-mode random
```

Tests separate copied bits, ordered-gradient bits, independent numerical
accuracy and invariance. Coverage includes ranks 1–4, positive/negative axes,
H3 and BAGEL layouts, strides, offsets, expanded tensors, empty axes, repeated
indices, cancellation, realistic H3 row counts, nondefault streams, and a
two-GPU device-restoration regression. Small-sort size and integer-overflow
boundaries, mixed 4/5, 16/17, 128/129 and 256/257 occurrence groups, overflow
queues, workspace-cap fallback, grouping-path equivalence and signed-zero
gradients are also covered. Invalid-index checks remain enabled with
`TRITON_DEBUG=0`, including empty output dimensions. Existing WS1 tolerances
are unchanged.
The checker defaults to `[B,S,D]`, `dim=1` when no explicit selection shape is
given; `--candidate pytorch --device cpu` runs the reference locally.

The final shared implementation was validated on one NVIDIA H100 NVL with
Python 3.12.3, PyTorch 2.8.0+cu128 and Triton 3.4.0: 163 operator/benchmark
tests passed and 4 two-GPU cases were skipped. The registry/input suite added
5 passes, and the BF16 unified checker above passed with random upstream
gradients. These runs include contiguous/strided forward and direct, dense-bucket
and sorted backward paths. Two-GPU restoration remains pending for this version;
single-GPU results do not qualify it or other architectures.

## Benchmark

```bash
export PYTHONPATH=.
python benchmarks/operators/embedding/benchmark_index_select.py --dim 1 --dry-run
# H3-shaped output selection
timeout 1200 python benchmarks/operators/embedding/benchmark_index_select.py \
  --shapes 1x129x32 1x38734x96 1x110750x96 --dim 1 \
  --duplicates 0 0.25 --dtype float32 --output reports/index-select/h3.json
# BAGEL-shaped packed hidden/query selection (synthetic inputs)
timeout 1200 python benchmarks/operators/embedding/benchmark_index_select.py \
  --shapes 129x3584 4097x28x128 --dim 0 \
  --duplicates 0 0.25 --dtype bfloat16 --output reports/index-select/bagel.json
# H100 native and actual registry dispatch, including CUDA Graph replay
timeout 1200 python benchmarks/operators/embedding/benchmark_index_select.py \
  --shapes 1x129x32 1x38734x96 1x110750x32 --dim 1 \
  --duplicates 0 0.25 --dtype float32 --mode both \
  --providers torch torch_deterministic triton cuda auto \
  --output reports/index-select/h100.json
```

`timeout` above is GNU timeout on the GPU host. Shapes, selected axis and
duplicate fraction are explicit; repeat with FP16/FP32/BF16 as needed. Inputs
select about half the axis, with synthetic duplicate distributions.

Compare eager `torch.index_select` + autograd, its deterministic-mode variant,
and the ordered Triton operator. `cuda` explicitly selects the H100 native
backend; `auto` measures the real registry dispatcher. Time forward, backward and forward+backward,
including allocation, validation, wrappers, zero fill and backward index grouping.
Backward reuses a retained graph; combined calls construct a fresh graph.
Input creation, reference checks, compilation and warmup are outside timing.
CUDA-event latency includes dispatch gaps; wall latency also includes end
synchronization. Incremental peak allocated memory measures the combined call,
excluding inputs and the already-retained graph.

Before timing, compare all elements with the independent accuracy oracle;
Triton, native CUDA and registry dispatch must also match the ordered FP32 VJP
bitwise and repeat exactly. All providers retain their accuracy, repeatability
and order mismatches in the report, including PyTorch baselines. Failed
candidate checks stop timing and retain an incomplete report. JSON includes
raw samples, medians, memory, source hashes, environment and completeness.

### Shared Triton baseline

An earlier H100 NVL diagnostic on 2026-11-13 covered 13 shapes/layouts in FP32, FP16
and BF16 (39 input combinations). The previous ordered implementation is the
local snapshot at the beginning of this H100 run, already containing earlier
A40 optimizations. Source fingerprints and raw measurements are retained in
the external evidence archive; the numbers below are from the final candidate.

Fixed-input CUDA Graph replay isolates captured device work from Python dispatch.
It includes copy, bounds-assertion, zero-fill and grouping kernels; host allocation,
autograd dispatch and compilation are excluded from replay. Forward runs with
autograd enabled, backward reuses a retained graph, and combined calls build a
fresh graph during capture. Two provider-order-reversed rounds each take five
samples of ten replays, with four calls per replay. Numbers are the mean of two
round medians. This diagnostic does not qualify arbitrary application Graph
integration.

Representative FP32 device times, **forward / backward / combined**, in microseconds:

| Input shape | dim | K | Duplicate fraction | Previous ordered | Current ordered | Ordinary PyTorch |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `1x129x32` | 1 | 64 | 0.25 | 1.92 / 2.26 / 3.60 | 1.90 / 2.37 / 3.76 | 2.04 / 3.35 / 5.06 |
| `1x4097x96` | 1 | 2048 | 0.25 | 2.20 / 8.78 / 10.55 | 2.33 / 6.66 / 8.47 | 3.16 / 5.34 / 7.93 |
| `1x38734x96` | 1 | 19367 | 0 | 5.16 / 18.05 / 24.10 | 4.32 / 12.36 / 18.11 | 14.64 / 19.75 / 34.84 |
| `1x110750x32` | 1 | 55375 | 0.25 | 4.27 / 31.43 / 36.45 | 4.03 / 16.17 / 21.72 | 38.53 / 19.40 / 57.80 |
| `1x367986x96` | 1 | 183993 | 0 | 51.86 / 102.17 / 153.59 | 45.42 / 89.75 / 134.09 | 124.15 / 190.32 / 313.49 |
| `128x4097` | 1 | 2048 | 0.25 | 3.86 / 12.75 / 16.06 | 2.95 / 11.25 / 13.71 | 5.93 / 7.57 / 13.10 |
| `1x2049x32` | 1 | 1024 | 0.99 | 2.09 / 22.77 / 24.35 | 2.17 / 15.08 / 16.79 | 2.52 / 4.25 / 6.23 |
| `1x32769x32` | 1 | 16384 | 0.9 | 2.55 / 67.71 / 69.82 | 2.55 / 12.79 / 14.88 | 12.64 / 7.81 / 20.11 |
| `1x131073x96` | 1 | 65536 | 0.99 | 13.08 / 203.56 / 216.24 | 9.56 / 53.23 / 68.78 | 45.59 / 67.13 / 113.28 |

Across 39 combinations, median device speedups over the previous ordered version
were 1.07x forward, 1.51x backward and 1.45x combined. Current device time was
below ordinary PyTorch in 37/39 forward, 21/39 backward and 33/39 combined comparisons.
Near-ties and these counts are descriptive, without statistical significance
claims. Narrow slices and long duplicate chains still have backward gaps.

**Public eager performance remains an open issue.** Separate measurements include
all allocation, validation, wrappers and Python/autograd dispatch: two reversed
rounds, five warmups and 30 samples per phase. Median wall speedups over the
previous version were 1.10x / 1.35x / 1.25x (forward/backward/combined); only
7/39 / 2/39 / 1/39 comparisons were faster than ordinary PyTorch. The rented host
showed substantial variation, so raw rounds remain in the report. Most dense
cases now use three launches instead of five, but cached-JIT and autograd dispatch
still cost more than native C++ calls. Device-only gains must not be presented as
eager speedups; no compilation was included in either timed measurement.

Ordinary PyTorch exceeded gradient accuracy tolerances in 5 input combinations
and had bitwise gradient repeatability failures in 9 combinations. PyTorch
deterministic mode exceeded gradient accuracy tolerances in 2 input combinations
and had no bitwise repeatability failures. Both ordered implementations
passed every benchmark accuracy, fixed-order and repeat check, without tolerance
changes. Native baselines do not implement the same occurrence-ordered FP32
contract: failures and timing are both retained. These observations do not prove
that the ordered implementation is universally more accurate.

Peak allocated memory and full per-dtype timing tables accompany the evidence.
Position slots trade storage for parallel metadata access: for FP32
`[1,131073,96]`, duplicate fraction 0.99, combined extra allocation is
137.30 MiB, versus 75.55 MiB before and 72.00 MiB for ordinary PyTorch.
The 128 MiB dense-metadata cap limits this tradeoff; it does not cap all
operator memory or stable-sort workspace. This optimization is not a universal
memory reduction. Keep raw reports and plots as external PR attachments rather
than committed source.

### H100 NVL native comparison

The native run used the same 13 shapes/distributions across FP32, FP16 and BF16,
with Python 3.12.3, PyTorch 2.8.0+cu128 and Triton 3.4.0. The final GPU suite
passed 285 tests; 8 two-GPU cases and 10 Triton-only helper cases were skipped.
Six unified-checker runs covered both H3/BAGEL axes and all three dtypes.
Native CUDA, shared Triton and actual registry dispatch passed every benchmark
accuracy, occurrence-order and repeat check. Ordinary/deterministic PyTorch
retained 5/2 accuracy-failing cases and 9/0 repeat-failing cases.

Measured geometric-mean speedup over shared Triton, **forward / backward / combined**
(ratios above one mean faster; two reversed rounds using the method above):

| Entry | Eager wall | CUDA Graph device |
| --- | --- | --- |
| Explicit native CUDA | 2.13 / 2.62 / 2.91 | 0.95 / 0.79 / 0.80 |
| Actual registry dispatch | 1.77 / 2.58 / 2.87 | 1.00 / 0.79 / 1.00 |

Native combined eager was faster than Triton in 39/39 cases, but only 8/39
against ordinary PyTorch; actual registry dispatch was faster than ordinary
PyTorch in 11/39. Host noise remained substantial: native combined round
medians differed by up to 6.31x. These observations do not establish universal
or full-model speedups. Native combined Graph replay beat Triton in only 9/39
cases, which motivates retaining Triton for forward calls inside capture.
Backward-only capture retains its earlier eager forward's native autograd node.

Extra peak allocation also varies by shape. FP32 `[1,367986,96]`, dim=1,
K=183993, no duplicates used 249.23 MiB native versus 212.03 MiB shared
(+37.20 MiB). FP32 `[65537]`, dim=0, K=32768, duplicate fraction 0.25 used
1.65 versus 33.03 MiB. The 128 MiB native limit covers integer workspace,
not outputs or total extra allocation. Full shapes, samples, baseline failures,
source hashes and memory measurements remain in the external evidence archive.
