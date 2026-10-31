# H100 ordered-reduction results

The [ordered-reduction helpers](../architecture/ordered-reduction.md) preserve a
fixed arithmetic tree in ordinary Triton and optional Gluon enclosing kernels.
The benchmark compares them with **ordinary `torch.sum`** and **native `tl.sum`**;
the explicit same-tree PyTorch implementation is a correctness reference, not a
speedup baseline.

This page reports the final H100 confirmation at
`1a2063b595f2154b677238911e74352ce1a95835`, also reported in
[PR #595](https://github.com/RL-Align/RL-Kernel/pull/595). It supersedes the earlier
tuning snapshot. All counts and timings below describe that measured revision,
not a GPU rerun of subsequent changes.

## Environment and scope

- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0, 132 SMs; PCI bus `00000000:BB:00.0`.
- NVIDIA driver: 595.91.07; PyTorch CUDA runtime: 13.0.
- Python 3.12.3, PyTorch 2.13.0+cu130, Triton 3.7.1.
- Run completed on 2026-10-30 at 16:14:46 UTC (2026-10-31 in Asia/Shanghai).

The run covered **39 shape/dtype cases and 159 provider measurements**. Inputs
are FP16/BF16/FP32; all providers accumulate and output FP32. Arithmetic tiles
contain 1024 elements, with initial groups of 8. The reduction operates on the
final dimension of `[rows, length]`. Cases include small and intermediate
widths, non-power-of-two tails, large matrices and physical suffix padding.

Provider settings were frozen from earlier H100 tuning before this confirmation;
there was no parameter search during the run. The comparisons below use the
preselected parallel Gluon caller for `[16,262144]`, and row-only Gluon elsewhere.
The extra row-only Gluon measurements for the long-row cases remain in the raw
report. This is not an automatic library dispatch policy or a universally optimal
configuration.

Product and custom callbacks passed correctness tests; their performance was not
tuned in this run. Results do not establish ROCm, other GPU models, arbitrary
fusion, full-model training/rollout parity, or cross-compiler reproducibility.

## Measurement method and correctness

These numbers are **CUDA Graph kernel timings**. Inputs, outputs and scratch are
preallocated. Compilation, input creation, accuracy checks and Graph capture are
excluded. Providers share at least 250 ms of warmup and run in shuffled order;
inputs are reused without explicit cache eviction. No `torch.compile` is used.
Both stages of a parallel implementation are included. Eager host dispatch and
allocation costs are not represented by these figures.

Each provider uses six rounds of 40 samples, with 64 invocations per replay for
small/medium cases and 16 for `[1024,262144]`. Reported latency is the median of
the round medians. Throughput inputs are uniform in `[0,1)`, seed 314159;
separate tests cover cancellation and other numerical stress cases. Every
comparison within a table row comes from this same confirmation run. No
measurement's maximum/minimum round-median ratio exceeded 1.15.

Before timing, candidates passed independent FP64 accuracy, sampled fixed-tree
PyTorch bitwise checks and all-row equality across ordered schedules. Graph
replay was checked against the eagerly launched result. The full H100 regression
at the measured revision passed **2562 tests, with zero skips, failures or
errors**, covering references, Triton/Gluon helpers and benchmark tests.

## Representative results

Latency in microseconds; **lower is better**. The native Triton provider and
ordered execution settings for each case are recorded in the frozen plan.
Long-row Gluon uses tile producers followed by the ordered merge; other Gluon
rows use one program per input row.

| Shape | Input | torch.sum | Native tl.sum | Ordered Triton | Ordered Gluon |
| --- | --- | ---: | ---: | ---: | ---: |
| 128×8193 | FP16 | 3.017 | 1.856 | 3.157 | 2.040 |
| 128×8193 | BF16 | 3.076 | 1.856 | 3.196 | 2.053 |
| 128×8193 | FP32 | 3.334 | 2.009 | 3.300 | 2.294 |
| 16×262144 | FP16 | 8.989 | 4.589 | 5.124 | 5.008 |
| 16×262144 | BF16 | 8.965 | 4.574 | 5.122 | 5.022 |
| 16×262144 | FP32 | 10.178 | 4.732 | 5.489 | 5.287 |
| 1024×262144 | FP16 | 174.754 | 177.344 | 178.551 | 186.751 |
| 1024×262144 | BF16 | 174.793 | 177.270 | 178.399 | 186.450 |
| 1024×262144 | FP32 | 345.174 | 348.223 | 348.358 | 344.869 |

For `[128,8193]`, Gluon latency exceeds native Triton by 9.9% (FP16), 10.6%
(BF16) and 14.2% (FP32). Here and below, percent overhead means
`100 × (ordered latency / baseline latency - 1)`. On the largest matrices,
ordinary Triton stays close to native Triton, while Gluon ranges from about
1% faster to 5.3% slower. Gluon is not consistently faster for every small input.

Few rows and a very long reduction require enough independent programs. For
`[16,262144]`, a row-only Gluon launch creates just 16 programs; the parallel caller
creates 4096 fixed-tile producers. This changes the execution schedule without
changing the tree or merge order. The grid and measurements support this
explanation; instruction-level hardware counters were not used to attribute all
remaining latency to a particular instruction.

The [PR figures](https://github.com/RL-Align/RL-Kernel/pull/595) and attachment use
these same final measurements. The latency figure shows microseconds on a log
axis, where lower is better. The speedup figure uses **baseline latency divided
by ordered latency**: above 1 is faster, below 1 is slower. Each dtype is shown
separately.

## Acceptance and remaining overhead

The agreed tolerance was no more than 10% slower than each native baseline,
allowing up to 0.5 µs absolute excess when that baseline is at most 10 µs. The
wider allowance applies. Each baseline is assessed separately.

| Implementation | Against torch.sum | Against native tl.sum |
| --- | ---: | ---: |
| Ordered Triton | 39/39 | 16/39 |
| Ordered Gluon | 39/39 | 37/39 |

Two Gluon cases remain outside the strict native-Triton gate:

| Input / shape | Overhead | Extra latency | Beyond the allowance |
| --- | ---: | ---: | ---: |
| FP32 128×16385 | +19.01% | +0.502375 µs | 0.002375 µs |
| FP32 16×262144 | +11.72% | +0.554750 µs | 0.054750 µs |

These near-boundary results were accepted for the current work, but remain
explicitly recorded as misses rather than counted as passing the strict gate.
Upper layers can keep ordinary Triton or choose a separately qualified Gluon
kernel. The library does not select a backend, promise these speedups for every
input, or substitute a different arithmetic tree for performance.

## Evidence and reproduction

Download the [H100 test and benchmark bundle](https://github.com/user-attachments/files/34192793/ordered-reduction-h100-final-20261031.zip).
Its ZIP SHA256 is
`c551e209dd9a21cbe41f4e045fa90bab2b5add9081f50897dac3633309fd1bc9`.
Measurements and archives are PR attachments, outside the source diff.

The bundle contains:

- `raw/results.json` and `raw/cases/*.json`: all raw timing samples, per-round
  medians, accuracy results, source hashes and execution settings.
- `raw/frozen-plan.json`: the exact 39-case plan and preselected providers.
- `raw/pytest.log`, `raw/pytest.xml`, `raw/pytest.exit`: regression results.
- `raw/commit.txt`, `raw/worktree-status.txt`, `raw/environment.txt`,
  `raw/nvidia-smi.txt`, `raw/packages.txt`: measured source and hardware/software identity.
- `latency.csv`, `summary.json`, the figures and `SHA256SUMS`: selected results,
  acceptance calculations and attachment checksums.
- `raw/run_final.py` and `raw/setup.sh`: the original run driver and environment setup.
- `historical/`: the earlier tuning snapshot, kept separately from the final run.

The driver uses fixed `/root` paths. To repeat the recorded run, prepare the
versions listed above using the package commands in `raw/setup.sh`, install the
repository at `/root/RL-Kernel-ordered-final`, and select the measured revision
explicitly. The original setup script's shallow branch clone now follows a newer
head; it must not be mistaken for a checkout of the measured commit.

From the extracted bundle directory, the reproduction command is:

```bash
git -C /root/RL-Kernel-ordered-final checkout --detach \
  1a2063b595f2154b677238911e74352ce1a95835
mkdir -p /root/ordered-reduction-final
cp raw/run_final.py raw/frozen-plan.json /root/ordered-reduction-final/
cd /root/RL-Kernel-ordered-final
/root/ordered-final-py/bin/python -u /root/ordered-reduction-final/run_final.py
```

The driver first runs the following regression with a 1200-second overall timeout
and a 90-second per-test timeout, then invokes the maintained benchmark worker
for each frozen case with a 180-second per-case timeout:

```bash
python -m pytest tests/reference/reduction tests/backends/shared/triton/reduction \
  tests/benchmarks/test_ordered_reduction_benchmark.py -q -rs -x \
  --timeout=90 --junitxml=/root/ordered-reduction-final/pytest.xml
```

The maintained benchmark CLI uses one explicit configuration per schedule. A
single invocation with its defaults does **not** reproduce the per-case settings
in this table; use the frozen plan and attachment driver for that comparison.
The earlier parameter-grid search and diagnostic candidates are not retained in
the source tree. New measurements can use the documented
[benchmark commands](../architecture/ordered-reduction.md#performance-comparison)
with explicit settings and keep their results separate from this recorded run.
