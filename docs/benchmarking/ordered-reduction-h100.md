# H100 ordered-reduction results

The [ordered-reduction helpers](../architecture/ordered-reduction.md) preserve a
fixed arithmetic tree in ordinary Triton and optional Gluon enclosing kernels.
The benchmark compares them with **ordinary `torch.sum`** and **native `tl.sum`**;
the explicit same-tree PyTorch implementation is a correctness reference, not a
speedup baseline.

This page reports the post-review H100 regression at
`4fd7469e7e273adb3bc82adbe547705c336caeec`, also reported in
[PR #595](https://github.com/RL-Align/RL-Kernel/pull/595). It supersedes the earlier
confirmation run. All counts and timings below describe that measured revision;
subsequent report-only updates do not change the measured kernels.

## Environment and scope

- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0, 132 SMs; PCI bus `00000000:65:00.0`.
- NVIDIA driver: 595.91.07; PyTorch CUDA runtime: 13.0.
- Python 3.12.3, PyTorch 2.13.0+cu130, Triton 3.7.1.
- Run completed on 2026-10-31 at 07:11:11 UTC (15:11:11 Asia/Shanghai).

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
at the measured revision passed **2580 tests, with zero skips, failures or
errors**, covering references, Triton/Gluon helpers and benchmark tests.

## Representative results

Latency in microseconds; **lower is better**. The native Triton provider and
ordered execution settings for each case are recorded in the frozen plan.
Long-row Gluon uses tile producers followed by the ordered merge; other Gluon
rows use one program per input row.

| Shape | Input | torch.sum | Native tl.sum | Ordered Triton | Ordered Gluon |
| --- | --- | ---: | ---: | ---: | ---: |
| 128×8193 | FP16 | 3.027 | 1.855 | 3.169 | 2.034 |
| 128×8193 | BF16 | 3.008 | 1.870 | 3.216 | 2.035 |
| 128×8193 | FP32 | 3.332 | 2.006 | 3.317 | 2.267 |
| 16×262144 | FP16 | 8.958 | 4.588 | 5.110 | 4.983 |
| 16×262144 | BF16 | 8.975 | 4.570 | 5.126 | 5.020 |
| 16×262144 | FP32 | 10.068 | 4.742 | 5.480 | 5.283 |
| 1024×262144 | FP16 | 171.620 | 173.295 | 173.158 | 172.051 |
| 1024×262144 | BF16 | 171.664 | 173.185 | 173.170 | 171.772 |
| 1024×262144 | FP32 | 338.087 | 338.964 | 337.432 | 335.560 |

For `[128,8193]`, Gluon latency exceeds native Triton by 9.7% (FP16), 8.9%
(BF16) and 13.0% (FP32). Here and below, percent overhead means
`100 × (ordered latency / baseline latency - 1)`. On the largest matrices,
ordinary Triton stays close to native Triton, while Gluon is about 0.7–1.0%
faster on this run. Gluon is not consistently faster for every small input.

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
| Ordered Gluon | 39/39 | 38/39 |

One Gluon case remains outside the strict native-Triton gate:

| Input / shape | Overhead | Extra latency | Beyond the allowance |
| --- | ---: | ---: | ---: |
| FP32 16×262144 | +11.39% | +0.540375 µs | 0.040375 µs |

This remains a near-boundary miss, rather than a strict pass. The other previous
boundary case, FP32 `[128,16385]`, falls inside the gate on this run; the changed
pass count is not evidence of a new kernel optimization. With the identical
frozen plan, the largest per-case latency increase from the preceding run was
1.76% for ordinary Triton and 1.83% for Gluon. The attachment records both runs'
comparison and all current measurements.

Upper layers can keep ordinary Triton or choose a separately qualified Gluon
kernel. The library does not select a backend, promise these speedups for every
input, or substitute a different arithmetic tree for performance.

## Evidence and reproduction

Download the [H100 test and benchmark bundle](https://github.com/user-attachments/files/34211949/ordered-reduction-h100-review-20261031.zip).
Its ZIP SHA256 is
`1a94ba54fba5e9519f480122117b5e30ba6e3915de044d997e4a4b5b4ea96b66`.
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
- `raw/run_final.py`, `raw/setup.sh`, `raw/run_all.sh` and `raw/reproduce.sh`: the
  measured driver, setup, timeout/packaging wrapper and pinned-revision replay.
- `comparison.json`: per-case changes from the preceding confirmation run, with
  both source revisions and unchanged provider settings.

The driver uses fixed `/root` paths. On a fresh H100 host with Git, Python 3.12
and uv installed, run the following from the extracted bundle directory:

```bash
bash raw/reproduce.sh
```

This fetches the measured commit directly, rather than following the moving
branch head, installs the recorded PyTorch/Triton versions, and then runs the
regression and frozen benchmark. It refuses to overwrite existing source or
result directories. Results are written to `/root/ordered-reduction-review` and
packaged as `/root/ordered-reduction-review-20261031.tar.gz`.

The driver first runs the following regression with a 1200-second overall timeout
and a 90-second per-test timeout, then invokes the maintained benchmark worker
for each frozen case with a 180-second per-case timeout:

```bash
python -m pytest tests/reference/reduction tests/backends/shared/triton/reduction \
  tests/benchmarks/test_ordered_reduction_benchmark.py -q -rs -x \
  --timeout=90 --junitxml=/root/ordered-reduction-review/pytest.xml
```

The maintained benchmark CLI uses one explicit configuration per schedule. A
single invocation with its defaults does **not** reproduce the per-case settings
in this table; use the frozen plan and attachment driver for that comparison.
The earlier parameter-grid search and diagnostic candidates are not retained in
the source tree. New measurements can use the documented
[benchmark commands](../architecture/ordered-reduction.md#performance-comparison)
with explicit settings and keep their results separate from this recorded run.
