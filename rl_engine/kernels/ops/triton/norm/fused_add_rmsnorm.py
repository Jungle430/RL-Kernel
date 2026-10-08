# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""FP32 arithmetic for fused residual addition and RMSNorm.

The caller supplies contiguous x/residual tensors viewed as [rows, n_cols],
a contiguous weight vector [n_cols], and FP32 y/updated_residual output buffers.
Forward also saves one FP32 inverse_rms per row for backward.
Launch one program per row, with a fixed power-of-two BLOCK_SIZE >= n_cols > 0,
fixed num_warps, and enable_fp_fusion=False.

Backward launches one program per row, reusing updated_residual and inverse_rms.
It writes input gradients and FP32 weight-gradient contributions [rows, n_cols].
A second kernel defaults to a left-fold in ascending row order. TILED uses row
lanes within each program. PARALLEL partitions rows across programs and merges
FP32 partials in a second launch. Both change addition order and are opt-in.
Both upstream gradient buffers are required; supply zeros for an unused output branch.
Gradient output buffers select the final storage dtypes. Use the same stream
for all backward launches, and disable FP fusion for all kernels.

Normalization uses the unrounded FP32 residual sum. Both forward outputs are
FP32; backward computes in FP32 and casts each input gradient to that input's
dtype on store. Model integration must preserve these declared cast points.
This operator is not yet registered as the model's strict implementation.
"""

import math
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch
import triton
import triton.language as tl

# ROCm PyTorch also exposes its devices through the CUDA namespace.
_SUPPORTED_DEVICES = ("cuda", "hip", "xpu", "musa")
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_NUM_WARPS = 4
_WEIGHT_BLOCK_ROWS = 32


class RMSNormWeightGradStrategy(Enum):
    """Select how FP32 per-row weight-gradient contributions are combined."""

    SEQUENTIAL = "sequential"
    TILED = "tiled"
    PARALLEL = "parallel"


@dataclass(frozen=True)
class RMSNormWeightGradConfig:
    """Explicit experimental configuration; no timing-based dispatch."""

    block_cols: int = 128  # Features owned by one program (power of two).
    block_rows: int = 32  # TILED row lanes / PARALLEL rows per partial; unused by SEQUENTIAL.
    num_warps: int = 4  # Warps per weight-reduction program, including the parallel merge.

    def __post_init__(self):
        for value in (self.block_cols, self.block_rows):
            if value <= 0 or value & (value - 1):
                raise ValueError("block_cols and block_rows must be positive powers of two.")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError("num_warps must be 1, 2, 4, or 8.")


_WEIGHT_GRAD_CONFIGS = {
    RMSNormWeightGradStrategy.SEQUENTIAL: RMSNormWeightGradConfig(block_rows=1),
    RMSNormWeightGradStrategy.TILED: RMSNormWeightGradConfig(),
    RMSNormWeightGradStrategy.PARALLEL: RMSNormWeightGradConfig(block_rows=256),
}


class _WeightGradLauncher(Protocol):
    """Reduce row contributions on the caller's current device and stream."""

    def __call__(
        self,
        *,
        grad_weight_per_row: torch.Tensor,
        grad_weight: torch.Tensor,
        n_rows: int,
        n_cols: int,
        config: RMSNormWeightGradConfig | None = None,
    ) -> None:
        """Write the reduction into a preallocated output.

        Args:
            grad_weight_per_row: Contiguous FP32 input with shape [n_rows, n_cols].
            grad_weight: Output buffer with shape [n_cols] and the weight dtype.
            n_rows: Number of input rows whose contributions are accumulated.
            n_cols: Number of features, equal to the number of weight elements.
            config: Explicit tile/warp settings, or this strategy's default.
        """
        ...


@triton.jit
def _fused_add_rmsnorm_fwd_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    y_ptr,
    updated_residual_ptr,
    inverse_rms_ptr,
    n_cols: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute u = x + residual and y = u * rsqrt(mean(u**2) + EPS) * weight."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    offsets = row * n_cols + cols
    mask = cols < n_cols

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # 1. Add the residual, keeping the intermediate sum in FP32.
    updated_residual = x + residual

    # 2. Reduce across this row only; masked columns contribute zero.
    sum_squares = tl.sum(updated_residual * updated_residual, axis=0, keep_dims=True)
    mean_square = tl.div_rn(sum_squares, n_cols)
    inverse_rms = tl.rsqrt(mean_square + EPS)

    # 3. Normalize each element, then apply its feature weight.
    normalized = updated_residual * inverse_rms
    y = normalized * weight

    tl.store(y_ptr + offsets, y, mask=mask)
    tl.store(updated_residual_ptr + offsets, updated_residual, mask=mask)
    tl.store(inverse_rms_ptr + row, tl.reshape(inverse_rms, ()))


@triton.jit
def _fused_add_rmsnorm_bwd_kernel(
    updated_residual_ptr,
    inverse_rms_ptr,
    weight_ptr,
    grad_y_ptr,
    grad_updated_residual_output_ptr,
    grad_x_ptr,
    grad_residual_ptr,
    grad_weight_per_row_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute input gradients and each row's FP32 weight-gradient contribution."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    offsets = row * n_cols + cols
    mask = cols < n_cols

    updated_residual = tl.load(updated_residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inverse_rms = tl.load(inverse_rms_ptr + row).to(tl.float32)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    grad_y = tl.load(grad_y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_updated_residual_output = tl.load(
        grad_updated_residual_output_ptr + offsets, mask=mask, other=0.0
    ).to(tl.float32)

    # 1. Reuse the FP32 sum and row statistic saved by forward.
    normalized = updated_residual * inverse_rms

    # 2. y = normalized * weight. Save this row's weight-gradient contribution.
    grad_normalized = grad_y * weight
    grad_weight_per_row = grad_y * normalized

    # 3. RMSNorm backward: rstd * (g - normalized * mean(g * normalized)).
    # One fixed row reduction combines the direct and inverse_rms paths.
    correction_sum = tl.sum(grad_normalized * normalized, axis=0)
    correction = tl.div_rn(correction_sum, n_cols)
    grad_updated_residual_from_y = inverse_rms * (grad_normalized - normalized * correction)

    # 4. Add the gradient from the separate residual output.
    grad_updated_residual_total = grad_updated_residual_from_y + grad_updated_residual_output

    # 5. updated_residual = x + residual: both inputs receive this gradient.
    grad_x = grad_updated_residual_total
    grad_residual = grad_updated_residual_total

    tl.store(grad_x_ptr + offsets, grad_x.to(grad_x_ptr.dtype.element_ty), mask=mask)
    tl.store(
        grad_residual_ptr + offsets,
        grad_residual.to(grad_residual_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(grad_weight_per_row_ptr + offsets, grad_weight_per_row, mask=mask)


@triton.jit
def _fused_add_rmsnorm_bwd_weight_kernel(
    grad_weight_per_row_ptr,
    grad_weight_ptr,
    n_rows,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Left-fold FP32 row contributions; launch ceil(n_cols / BLOCK_SIZE) programs.

    Each program owns a block of columns. Rows are accumulated sequentially,
    with no atomics or row-count-dependent reduction tree. Zero rows yield zero.
    """
    block = tl.program_id(0).to(tl.int64)
    cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    grad_weight = tl.full((BLOCK_SIZE,), 0.0, tl.float32)
    for row in range(n_rows):
        offsets = row.to(tl.int64) * n_cols + cols
        contribution = tl.load(grad_weight_per_row_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        grad_weight = grad_weight + contribution

    tl.store(grad_weight_ptr + cols, grad_weight.to(grad_weight_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fused_add_rmsnorm_bwd_weight_tiled_kernel(
    grad_weight_per_row_ptr,
    grad_weight_ptr,
    n_rows,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Experiment: accumulate row tiles, then reduce the row lanes once.

    Each program still owns distinct output columns; no atomics are used.
    Row lane k accumulates rows k, k + BLOCK_ROWS, ... in FP32. Reducing those
    lanes changes rounding relative to the sequential kernel, even with FP
    fusion disabled. Zero rows yield zero; both row and column tails are masked.
    """
    block = tl.program_id(0).to(tl.int64)
    cols = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rows = tl.arange(0, BLOCK_ROWS).to(tl.int64)

    grad_weight = tl.full((BLOCK_ROWS, BLOCK_SIZE), 0.0, tl.float32)
    for start in range(0, n_rows, BLOCK_ROWS):
        current_rows = start + rows
        offsets = current_rows[:, None] * n_cols + cols[None, :]
        mask = (current_rows[:, None] < n_rows) & (cols[None, :] < n_cols)
        contribution = tl.load(grad_weight_per_row_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        grad_weight = grad_weight + contribution

    grad_weight_sum = tl.sum(grad_weight, axis=0)
    tl.store(
        grad_weight_ptr + cols,
        grad_weight_sum.to(grad_weight_ptr.dtype.element_ty),
        mask=cols < n_cols,
    )


def _launch_fused_add_rmsnorm_bwd_weight_sequential(
    *,
    grad_weight_per_row: torch.Tensor,
    grad_weight: torch.Tensor,
    n_rows: int,
    n_cols: int,
    config: RMSNormWeightGradConfig | None = None,
) -> None:
    """Write weight gradients with the sequential kernel's launch configuration."""
    config = config or _WEIGHT_GRAD_CONFIGS[RMSNormWeightGradStrategy.SEQUENTIAL]
    grid = (triton.cdiv(n_cols, config.block_cols),)
    _fused_add_rmsnorm_bwd_weight_kernel[grid](
        grad_weight_per_row,
        grad_weight,
        n_rows,
        n_cols,
        BLOCK_SIZE=config.block_cols,
        num_warps=config.num_warps,
        enable_fp_fusion=False,
    )


def _launch_fused_add_rmsnorm_bwd_weight_tiled(
    *,
    grad_weight_per_row: torch.Tensor,
    grad_weight: torch.Tensor,
    n_rows: int,
    n_cols: int,
    config: RMSNormWeightGradConfig | None = None,
) -> None:
    """Write weight gradients with the tiled kernel's launch configuration."""
    config = config or _WEIGHT_GRAD_CONFIGS[RMSNormWeightGradStrategy.TILED]
    grid = (triton.cdiv(n_cols, config.block_cols),)
    _fused_add_rmsnorm_bwd_weight_tiled_kernel[grid](
        grad_weight_per_row,
        grad_weight,
        n_rows,
        n_cols,
        BLOCK_SIZE=config.block_cols,
        BLOCK_ROWS=config.block_rows,
        num_warps=config.num_warps,
        enable_fp_fusion=False,
    )


@triton.jit
def _fused_add_rmsnorm_bwd_weight_partial_kernel(
    contributions_ptr,
    partials_ptr,
    n_rows,
    n_cols: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """One independent FP32 sum per fixed row tile and column tile; no atomics."""
    col_block = tl.program_id(0).to(tl.int64)
    row_block = tl.program_id(1).to(tl.int64)
    cols = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    offsets = rows[:, None] * n_cols + cols[None, :]
    mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
    values = tl.load(contributions_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    partial = tl.sum(values, axis=0)
    tl.store(partials_ptr + row_block * n_cols + cols, partial, mask=cols < n_cols)


def _launch_fused_add_rmsnorm_bwd_weight_parallel(
    *,
    grad_weight_per_row: torch.Tensor,
    grad_weight: torch.Tensor,
    n_rows: int,
    n_cols: int,
    config: RMSNormWeightGradConfig | None = None,
    partials: torch.Tensor | None = None,
) -> None:
    """Partition rows, then merge on the same stream using a fixed reduction layout.

    Optional partials is an FP32 [max(1, ceil(rows / block_rows)), n_cols]
    workspace on the input device. Microbenchmarks preallocate it; public calls
    allocate it here. Every element is overwritten on every launch, including tails.
    The split depends on the explicit configuration, never on SM count/occupancy.
    """
    config = config or _WEIGHT_GRAD_CONFIGS[RMSNormWeightGradStrategy.PARALLEL]
    n_partials = max(1, triton.cdiv(n_rows, config.block_rows))
    if partials is None:
        partials = torch.empty(
            (n_partials, n_cols), device=grad_weight_per_row.device, dtype=torch.float32
        )
    col_blocks = triton.cdiv(n_cols, config.block_cols)
    _fused_add_rmsnorm_bwd_weight_partial_kernel[(col_blocks, n_partials)](
        grad_weight_per_row,
        partials,
        n_rows,
        n_cols,
        BLOCK_ROWS=config.block_rows,
        BLOCK_COLS=config.block_cols,
        num_warps=config.num_warps,
        enable_fp_fusion=False,
    )
    # Reuse a fixed 32-lane fold, now over partials rather than all input rows.
    _fused_add_rmsnorm_bwd_weight_tiled_kernel[(col_blocks,)](
        partials,
        grad_weight,
        n_partials,
        n_cols,
        BLOCK_SIZE=config.block_cols,
        BLOCK_ROWS=_WEIGHT_BLOCK_ROWS,
        num_warps=config.num_warps,
        enable_fp_fusion=False,
    )


_WEIGHT_GRAD_LAUNCHERS: dict[RMSNormWeightGradStrategy, _WeightGradLauncher] = {
    RMSNormWeightGradStrategy.SEQUENTIAL: _launch_fused_add_rmsnorm_bwd_weight_sequential,
    RMSNormWeightGradStrategy.TILED: _launch_fused_add_rmsnorm_bwd_weight_tiled,
    RMSNormWeightGradStrategy.PARALLEL: _launch_fused_add_rmsnorm_bwd_weight_parallel,
}


def _validate_inputs(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("x must have shape [..., D] with D > 0.")
    if residual.shape != x.shape:
        raise ValueError("residual must have the same shape as x.")
    if weight.shape != (x.shape[-1],):
        raise ValueError("weight must have shape [D].")
    if residual.device != x.device or weight.device != x.device:
        raise ValueError("x, residual, and weight must be on the same device.")
    if any(t.dtype not in _SUPPORTED_DTYPES for t in (x, residual, weight)):
        raise TypeError(f"x, residual, and weight must have dtype in {_SUPPORTED_DTYPES}.")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive.")
    if x.device.type not in _SUPPORTED_DEVICES:
        raise ValueError(f"Triton fused add RMSNorm requires a device in {_SUPPORTED_DEVICES}.")


def _validate_backward_inputs(
    updated_residual: torch.Tensor,
    inverse_rms: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    if updated_residual.ndim == 0 or updated_residual.shape[-1] == 0:
        raise ValueError("updated_residual must have shape [..., D] with D > 0.")
    if updated_residual.device.type not in _SUPPORTED_DEVICES:
        raise ValueError(f"Triton fused add RMSNorm requires a device in {_SUPPORTED_DEVICES}.")
    n_cols = updated_residual.shape[-1]
    if inverse_rms.shape != (updated_residual.numel() // n_cols,):
        raise ValueError("inverse_rms must have one value per flattened input row.")
    if weight.shape != (n_cols,) or weight.dtype not in _SUPPORTED_DTYPES:
        raise ValueError("weight must have shape [D] and a supported floating dtype.")
    if inverse_rms.device != updated_residual.device or weight.device != updated_residual.device:
        raise ValueError("saved tensors must be on the same device.")


def _launch_fused_add_rmsnorm_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return two FP32 outputs and the internal per-row FP32 inverse_rms cache."""
    _validate_inputs(x, residual, weight, eps)
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols
    y = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    updated_residual = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    inverse_rms = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    if n_rows == 0:
        return y, updated_residual, inverse_rms

    with torch.cuda.device(x.device) if x.device.type == "cuda" else nullcontext():
        _fused_add_rmsnorm_fwd_kernel[(n_rows,)](
            x.contiguous(),
            residual.contiguous(),
            weight.contiguous(),
            y,
            updated_residual,
            inverse_rms,
            n_cols,
            eps,
            BLOCK_SIZE=triton.next_power_of_2(n_cols),
            num_warps=_NUM_WARPS,
            enable_fp_fusion=False,
        )
    return y, updated_residual, inverse_rms


def _launch_fused_add_rmsnorm_bwd(
    updated_residual: torch.Tensor,
    inverse_rms: torch.Tensor,
    weight: torch.Tensor,
    grad_y: torch.Tensor | None,
    grad_updated_residual_output: torch.Tensor | None,
    *,
    x_dtype: torch.dtype,
    residual_dtype: torch.dtype,
    weight_grad_strategy: RMSNormWeightGradStrategy = RMSNormWeightGradStrategy.SEQUENTIAL,
    weight_grad_config: RMSNormWeightGradConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return input-dtype gradients using the FP32 values saved by forward.

    An unused output contributes zero. By default, weight gradients left-fold
    FP32 row contributions before casting to the weight dtype. TILED/PARALLEL
    opt into other summation orders. No strategy guarantees that adding
    separately reduced microbatch gradients reproduces a single call bitwise.
    """
    _validate_backward_inputs(updated_residual, inverse_rms, weight)
    n_cols = updated_residual.shape[-1]
    n_rows = updated_residual.numel() // n_cols
    grad_x = torch.empty(updated_residual.shape, device=updated_residual.device, dtype=x_dtype)
    grad_residual = torch.empty(
        updated_residual.shape, device=updated_residual.device, dtype=residual_dtype
    )
    if n_rows == 0:
        return grad_x, grad_residual, torch.zeros_like(weight)

    grad_y_c = torch.zeros_like(updated_residual) if grad_y is None else grad_y.contiguous()
    grad_updated_residual_c = (
        torch.zeros_like(updated_residual)
        if grad_updated_residual_output is None
        else grad_updated_residual_output.contiguous()
    )
    grad_weight = torch.empty((n_cols,), device=weight.device, dtype=weight.dtype)
    grad_weight_per_row = torch.empty(
        (n_rows, n_cols), device=updated_residual.device, dtype=torch.float32
    )
    with (
        torch.cuda.device(updated_residual.device)
        if updated_residual.device.type == "cuda"
        else nullcontext()
    ):
        _fused_add_rmsnorm_bwd_kernel[(n_rows,)](
            updated_residual.contiguous(),
            inverse_rms.contiguous(),
            weight.contiguous(),
            grad_y_c,
            grad_updated_residual_c,
            grad_x,
            grad_residual,
            grad_weight_per_row,
            n_cols,
            BLOCK_SIZE=triton.next_power_of_2(n_cols),
            num_warps=_NUM_WARPS,
            enable_fp_fusion=False,
        )
        # Launch on the same stream: the merge observes completed row contributions.
        launch_weight_grad = _WEIGHT_GRAD_LAUNCHERS[weight_grad_strategy]
        launch_weight_grad(
            grad_weight_per_row=grad_weight_per_row,
            grad_weight=grad_weight,
            n_rows=n_rows,
            n_cols=n_cols,
            config=weight_grad_config,
        )
    return grad_x, grad_residual, grad_weight


class _FusedAddRMSNormTritonFunction(torch.autograd.Function):
    """Connect the two outputs and three input gradients to PyTorch autograd."""

    @staticmethod
    def forward(ctx, x, residual, weight, eps, weight_grad_strategy, weight_grad_config):
        y, updated_residual, inverse_rms = _launch_fused_add_rmsnorm_fwd(
            x, residual, weight, eps=eps
        )
        # Keep the existing FP32 output, not copies of x/residual or a rounded sum.
        ctx.save_for_backward(updated_residual, inverse_rms, weight)
        ctx.x_dtype = x.dtype
        ctx.residual_dtype = residual.dtype
        ctx.weight_grad_strategy = weight_grad_strategy
        ctx.weight_grad_config = weight_grad_config
        # Backward explicitly handles an output that was not used by the loss.
        ctx.set_materialize_grads(False)
        return y, updated_residual

    @staticmethod
    def backward(ctx, grad_y, grad_updated_residual_output):
        if grad_y is None and grad_updated_residual_output is None:
            return None, None, None, None, None, None
        updated_residual, inverse_rms, weight = ctx.saved_tensors
        gradients = _launch_fused_add_rmsnorm_bwd(
            updated_residual,
            inverse_rms,
            weight,
            grad_y,
            grad_updated_residual_output,
            x_dtype=ctx.x_dtype,
            residual_dtype=ctx.residual_dtype,
            weight_grad_strategy=ctx.weight_grad_strategy,
            weight_grad_config=ctx.weight_grad_config,
        )
        grad_x, grad_residual, grad_weight = (
            gradient if needed else None
            for gradient, needed in zip(gradients, ctx.needs_input_grad[:3], strict=True)
        )
        return grad_x, grad_residual, grad_weight, None, None, None


class TritonFusedAddRMSNormOp:
    """FP32-output fused add RMSNorm with first-order autograd on accepted GPUs.

    x/residual have identical shape [..., D], and weight has shape [D]. All
    inputs share a device and may independently use FP16, BF16, or FP32.
    Both outputs retain the input shape and use FP32; each input gradient uses
    that input's dtype. Noncontiguous tensors are copied to contiguous buffers.
    Normalization uses the FP32 residual sum without an intermediate downcast.
    weight_grad_strategy selects only the final weight-gradient reduction.
    SEQUENTIAL preserves the original order; TILED/PARALLEL are opt-in experiments.
    weight_grad_config optionally overrides the reduction tile/warp settings.
    """

    op_class = "norm"

    def __init__(
        self,
        *,
        weight_grad_strategy: RMSNormWeightGradStrategy = RMSNormWeightGradStrategy.SEQUENTIAL,
        weight_grad_config: RMSNormWeightGradConfig | None = None,
    ):
        self.weight_grad_strategy = weight_grad_strategy
        self.weight_grad_config = weight_grad_config

    def __call__(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(x, residual, weight, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _FusedAddRMSNormTritonFunction.apply(
            x, residual, weight, eps, self.weight_grad_strategy, self.weight_grad_config
        )
