# H100 ordered-reduction results

The [ordered-reduction helpers](../architecture/ordered-reduction.md) preserve a
fixed arithmetic tree in ordinary Triton and optional Gluon enclosing kernels.
The benchmark compares them with **ordinary `torch.sum`** and **native `tl.sum`**;
the explicit same-tree PyTorch implementation is a correctness reference, not a
speedup baseline.

## Environment and scope

NVIDIA H100 80GB HBM3 (132 SM), PyTorch 2.13.0+cu130, Triton 3.7.1, Python 3.12.3;
measurements collected on 2026-10-30. Inputs are FP16/BF16/FP32; all providers use
FP32 accumulation and output. Arithmetic tiles contain 1024 elements, with initial
groups of 8. The reduction operates on the final dimension of `[rows, length]`.

The broad run covered 39 shape/dtype cases and 1371 provider measurements. It
included small and intermediate widths, non-power-of-two tails, large matrices
and physical suffix padding. A subsequent three-case confirmation targeted
`[16,262144]`, including Gluon producers distributed across tiles and a separate
ordered merge. It did not rerun the full 39-case matrix.

Product and custom callbacks passed correctness tests; their performance was not
tuned in this run. Results do not establish ROCm, other GPU models, arbitrary
fusion, full-model training/rollout parity, or cross-compiler reproducibility.

## Measurement method

These numbers are **CUDA Graph kernel timings**. Inputs, outputs and scratch are
preallocated. Compilation, input creation, accuracy checks and Graph capture are
excluded. Providers share at least 250 ms of warmup and run in shuffled order;
inputs are reused without explicit cache eviction. No `torch.compile` is used.
Both stages of a parallel implementation are included. Eager host dispatch and
allocation costs are not represented by these figures.

The broad run used four rounds of 25 samples, with 128 invocations per replay for
small/medium cases and 8 for large matrices. Its table retains the fastest measured
configuration in each family and case. This is a tuning result, **not** an automatic
library dispatch policy or a single parameter set that is fastest everywhere.

The long-row confirmation froze the screening winners, changed the input seed to
1716, and used six rounds of 50 samples with 64 invocations per replay. It performed
no configuration search during confirmation. Reported latency is the median of
the round medians. Each comparison within a table row comes from the same run.

Before timing, candidates passed independent FP64 accuracy, sampled fixed-tree
PyTorch bitwise checks and all-row equality across ordered schedules. Graph
replay was checked against the eagerly launched result. The latest H100 test run
passed 157 tests, including 142 backend checks and 15 benchmark checks. This
count describes the measured snapshot before file organization and benchmark
cleanup, not a new GPU run of the final namespace layout.

## Representative results

Latency in microseconds; lower is better. Native Triton is the faster measured
full-row or two-stage `tl.sum` implementation for that case. Long-row Gluon uses
tile producers followed by the ordered merge; other Gluon rows use one program
per input row.

| Shape | Input | torch.sum | Native tl.sum | Ordered Triton | Ordered Gluon |
| --- | --- | ---: | ---: | ---: | ---: |
| 128×8193 | BF16 | 3.006 | 1.855 | 3.123 | 1.998 |
| 128×8193 | FP16 | 3.005 | 1.843 | 3.112 | 2.005 |
| 128×8193 | FP32 | 3.302 | 2.033 | 3.247 | 2.225 |
| 16×262144 | BF16 | 9.108 | 4.589 | 5.137 | 5.043 |
| 16×262144 | FP16 | 8.985 | 4.605 | 5.127 | 5.047 |
| 16×262144 | FP32 | 9.960 | 4.734 | 5.452 | 5.240 |
| 1024×262144 | BF16 | 171.948 | 173.436 | 173.318 | 169.354 |
| 1024×262144 | FP16 | 171.992 | 173.486 | 173.498 | 169.302 |
| 1024×262144 | FP32 | 338.382 | 339.350 | 337.888 | 336.036 |

Gluon's explicit layouts reduce the gap on medium widths. For `[128,8193]`, its
latency is 8–9% above native Triton, versus 60–69% for ordered ordinary Triton.
On the largest matrices, both ordered implementations track the baselines
closely. Very small inputs do not consistently
benefit from Gluon: FP16 `[3,1025]` took 1.198 µs with ordinary ordered Triton and
1.234 µs with Gluon.

Few rows and a very long reduction require enough independent programs. For
`[16,262144]`, a row-only Gluon launch creates just 16 programs; the parallel caller
creates 4096 fixed-tile producers. This changes the execution schedule without
changing the tree or merge order. The grid and measurements support this
explanation; instruction-level hardware counters were not used to attribute all
remaining latency to a particular instruction.

## Acceptance and remaining overhead

The agreed screening tolerance was no more than 10% slower than each native
baseline, allowing up to 0.5 µs absolute excess when that baseline is at most
10 µs. The wider allowance applies. In the broad run, ordinary ordered Triton
met this gate in 17/39 cases and the row-only Gluon version in 36/39. Per-case
selection between them met 37/39; this was not a production selector.

In the later long-row confirmation, Gluon parallel exceeded native Triton by
9.89% (BF16), 9.59% (FP16) and 10.71% (FP32). FP32's absolute excess was
0.506875 µs: **0.006875 µs beyond the strict allowance**. This near-boundary result
was accepted for the current work, but it is still recorded as outside the
strict gate. The follow-up is not a claim that a fresh full-matrix run passed.

Upper layers can keep ordinary Triton or choose a separately qualified Gluon
kernel. The reduction library does not select a backend, promise these speedups
for every input, or substitute a different arithmetic tree for performance.

## Evidence and reproduction

Measurement files and raw-sample archives are kept outside the source tree for
attachment to the [PR discussion](https://github.com/RL-Align/RL-Kernel/pull/595).
The evidence includes the 39 broad-case comparisons, all 15 confirmation
measurements, exact configurations, round medians, variation, errors, environment
and measured-source hashes. Its manifest records the raw archives' SHA256
identities. Historical source paths identify the measured snapshot before cleanup.
The evidence also records the mapping to the
earlier namespace cleanup: executable ASTs of the two implementations, reference,
metadata predicate and benchmark kernel modules matched after normalizing import
paths and docstrings. The subsequent shared-Gluon revision moves that implementation
to `backends/shared/triton/reduction/gluon.py` and parameterizes warp/wave width.
The H100 default remains 32; the measured arithmetic and layout choices are retained.
This report does not claim a new GPU run of that revision or AMD validation.

The maintained benchmark uses one explicit configuration per schedule. It keeps
accuracy/replay checks, timeout recovery and both native baselines, without
retaining the parameter-grid search and diagnostic candidates used during tuning.
A fresh run therefore reports the requested settings, not the historical table's
per-case winners. See the [benchmark commands](../architecture/ordered-reduction.md#performance-comparison).
