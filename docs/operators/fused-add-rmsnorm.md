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
| CUDA | `TritonFusedAddRMSNormOp` | preferred; GPU validation pending |
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
- `RMSNormWeightGradStrategy.SEQUENTIAL` is the default: each program owns 128
  features and folds row contributions in ascending order, without atomics.
- `RMSNormWeightGradStrategy.TILED` is experimental: each program accumulates
  32 row lanes by 128 features, then reduces the row lanes once. It changes the
  FP32 addition order. Benchmark results are needed before changing defaults.

Opt into the experiment explicitly:

```python
from rl_engine.kernels.ops.triton.norm import RMSNormWeightGradStrategy, TritonFusedAddRMSNormOp

op = TritonFusedAddRMSNormOp(weight_grad_strategy=RMSNormWeightGradStrategy.TILED)
```

The two strategies share forward and per-row backward arithmetic. Weight gradients
sum across rows and are checked numerically plus repeatably for an identical call;
they are not promised to match bitwise across strategies, row reorderings, or
separately reduced microbatches. Full-model, distributed, and CUDA-to-ROCm parity
require separate validation. FP32 intermediates alone do not prove these properties.

## Validation

The operator tests use an independent FP64 autograd reference, random upstreams
for both outputs, single-output losses, mixed dtypes, empty/strided inputs, unused
input gradients, repeated backward, saved-tensor immutability, masked tails, and
noncurrent-device launches on two GPUs. Default elementwise `(atol, rtol)` checks
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

## Benchmark

```bash
uv run --no-sync python benchmarks/benchmark_fused_add_rmsnorm.py --dry-run
uv run --no-sync python benchmarks/benchmark_fused_add_rmsnorm.py
```

Default coverage is 36 inputs: FP16/BF16/FP32, rows 1/32/33/128/1024/8192, and
widths 129/2688, with FP32 weights and upstreams. The benchmark compares eager
PyTorch with both Triton strategies for forward, backward, forward+backward,
and the isolated weight reduction (432 measurements). The 129-wide cases
exercise masked tails; width 2688 comes from the target model.

Outputs and random-upstream gradients are checked before each provider's timing.
For large row sums the benchmark records a weight-gradient absolute tolerance
of `2e-5 * sqrt(rows)` with relative tolerance `2e-5`, together with actual errors.
This diagnostic tolerance does not replace the model's strict acceptance gate.

Timing uses accelerator events, 10 warmups and 50 repetitions by default. There
is no `torch.compile`, CUDA Graph capture or explicit cache eviction. Compilation
and correctness checks are excluded. Public timings include allocation and
dispatch; small cases can include host dispatch gaps. Forward uses `no_grad`;
backward reuses a prebuilt graph; forward+backward builds a fresh graph each call.
The reduction microbenchmark reuses identical FP32 contributions and preallocated
outputs, comparing against `torch.sum`. Extra peak allocation excludes inputs and
prebuilt graphs. JSON includes samples, standard deviations, errors and memory;
Markdown reports latency and speedups relative to eager PyTorch.

Reports default to `reports/fused-add-rmsnorm/{report.md,results.json}`; use
`--output-dir` for each new run and attach reports to the PR instead of committing
them. No performance result has been established yet. Qualification against the
model's native production norm path and final cast contract remains pending.
