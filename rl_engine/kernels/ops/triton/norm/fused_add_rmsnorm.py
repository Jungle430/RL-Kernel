# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""FP32 arithmetic prototype for fused residual addition and RMSNorm.

The caller supplies contiguous x/residual tensors viewed as [rows, n_cols],
a contiguous weight vector [n_cols], and FP32 y/updated_residual output buffers.
Launch one program per row, with a fixed power-of-two BLOCK_SIZE >= n_cols > 0,
fixed num_warps, and enable_fp_fusion=False.

Backward also launches one program per row and recomputes the same statistics.
It writes input gradients and FP32 weight-gradient contributions [rows, n_cols].
A second kernel left-folds those contributions in ascending row order, matching
the accumulation order of the existing reduce_rows_fp32 helper. Both upstream
gradient buffers are required; supply zeros for an unused output branch.
Gradient output buffers select the final storage dtypes. Use the same stream
for both backward launches, and disable FP fusion for both kernels.

This prototype normalizes the unrounded FP32 residual sum. Input/output dtype
support and any BF16 residual rounding point still need a model-level contract.
The wrappers below expose the current FP32-output prototype explicitly; it is
not yet registered as the model's strict fused-add/RMSNorm implementation.
"""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from rl_engine.kernels.ops.pytorch.norm.fused_add_rmsnorm import (
    _SUPPORTED_DTYPES,
)
from rl_engine.kernels.ops.pytorch.norm.fused_add_rmsnorm import (
    _validate_inputs as _validate_tensor_inputs,
)

# ROCm PyTorch also exposes its devices through the CUDA namespace.
_SUPPORTED_DEVICES = ("cuda",)
_NUM_WARPS = 4
_WEIGHT_BLOCK_SIZE = 128


@triton.jit
def _fused_add_rmsnorm_fwd_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    y_ptr,
    updated_residual_ptr,
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

    # 1. Add the residual, keeping the intermediate sum in FP32 for now.
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


@triton.jit
def _fused_add_rmsnorm_bwd_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    grad_y_ptr,
    grad_updated_residual_output_ptr,
    grad_x_ptr,
    grad_residual_ptr,
    grad_weight_per_row_ptr,
    n_cols: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute input gradients and each row's FP32 weight-gradient contribution."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    offsets = row * n_cols + cols
    mask = cols < n_cols

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    grad_y = tl.load(grad_y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_updated_residual_output = tl.load(
        grad_updated_residual_output_ptr + offsets, mask=mask, other=0.0
    ).to(tl.float32)

    # Recompute the forward intermediates in the same order as the forward kernel.
    updated_residual = x + residual
    sum_squares = tl.sum(updated_residual * updated_residual, axis=0, keep_dims=True)
    mean_square = tl.div_rn(sum_squares, n_cols)
    inverse_rms = tl.rsqrt(mean_square + EPS)
    normalized = updated_residual * inverse_rms

    # 1. y = normalized * weight. Save this row's weight-gradient contribution.
    grad_normalized = grad_y * weight
    grad_weight_per_row = grad_y * normalized

    # 2. normalized = updated_residual * inverse_rms: follow both input paths.
    grad_updated_residual_direct = grad_normalized * inverse_rms
    grad_inverse_rms = tl.sum(grad_normalized * updated_residual, axis=0, keep_dims=True)

    # 3. inverse_rms = (mean_square + EPS) ** (-0.5).
    inverse_rms_cubed = inverse_rms * inverse_rms * inverse_rms
    grad_mean_square = grad_inverse_rms * (-0.5 * inverse_rms_cubed)

    # 4. mean_square = sum_squares / n_cols.
    grad_sum_squares = tl.div_rn(grad_mean_square, n_cols)

    # 5. The sum's backward broadcasts this per-row gradient over all columns.
    grad_squared = grad_sum_squares

    # 6. squared = updated_residual * updated_residual.
    grad_updated_residual_via_rms = grad_squared * (2.0 * updated_residual)

    # 7. Combine both paths from y and the separate residual-output gradient.
    grad_updated_residual_from_y = grad_updated_residual_direct + grad_updated_residual_via_rms
    grad_updated_residual_total = grad_updated_residual_from_y + grad_updated_residual_output

    # 8. updated_residual = x + residual: both inputs receive this gradient.
    grad_x = grad_updated_residual_total
    grad_residual = grad_updated_residual_total

    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)
    tl.store(grad_residual_ptr + offsets, grad_residual, mask=mask)
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
    offsets = cols
    for _ in range(n_rows):
        contribution = tl.load(grad_weight_per_row_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        grad_weight = grad_weight + contribution
        offsets = offsets + n_cols

    tl.store(grad_weight_ptr + cols, grad_weight, mask=mask)


def _validate_inputs(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    # Share metadata validation only; arithmetic stays in the Triton kernels.
    _validate_tensor_inputs(x, residual, weight, eps)
    if x.device.type not in _SUPPORTED_DEVICES:
        raise ValueError(f"Triton fused add RMSNorm requires a device in {_SUPPORTED_DEVICES}.")


def _validate_backward_inputs(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    grad_y: torch.Tensor | None,
    grad_updated_residual_output: torch.Tensor | None,
    eps: float,
) -> None:
    _validate_inputs(x, residual, weight, eps)
    for name, gradient in (
        ("grad_y", grad_y),
        ("grad_updated_residual_output", grad_updated_residual_output),
    ):
        if gradient is None:
            continue
        if gradient.shape != x.shape:
            raise ValueError(f"{name} must have the same shape as x.")
        if gradient.device != x.device:
            raise ValueError(f"{name} must be on the x device.")
        if gradient.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{name} must have dtype in {_SUPPORTED_DTYPES}.")


def _launch_fused_add_rmsnorm_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate FP32 outputs with the input shape and launch one program per row."""
    _validate_inputs(x, residual, weight, eps)
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols
    y = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    updated_residual = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    if n_rows == 0:
        return y, updated_residual

    _fused_add_rmsnorm_fwd_kernel[(n_rows,)](
        x.contiguous(),
        residual.contiguous(),
        weight.contiguous(),
        y,
        updated_residual,
        n_cols,
        eps,
        BLOCK_SIZE=triton.next_power_of_2(n_cols),
        num_warps=_NUM_WARPS,
        enable_fp_fusion=False,
    )
    return y, updated_residual


def _launch_fused_add_rmsnorm_bwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    grad_y: torch.Tensor | None,
    grad_updated_residual_output: torch.Tensor | None,
    *,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return input-dtype gradients, including both output branches.

    An unused output contributes zero. Weight gradients left-fold FP32 row
    contributions in flattened input order before casting to the weight dtype.
    This reduction describes one call; adding separately reduced microbatch
    gradients is not guaranteed to reproduce the same floating-point order.
    """
    _validate_backward_inputs(x, residual, weight, grad_y, grad_updated_residual_output, eps)
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols
    grad_x = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    grad_residual = torch.empty(residual.shape, device=residual.device, dtype=residual.dtype)
    if n_rows == 0:
        return grad_x, grad_residual, torch.zeros_like(weight)

    grad_y_c = (
        torch.zeros(x.shape, device=x.device, dtype=torch.float32)
        if grad_y is None
        else grad_y.contiguous()
    )
    grad_updated_residual_c = (
        torch.zeros(x.shape, device=x.device, dtype=torch.float32)
        if grad_updated_residual_output is None
        else grad_updated_residual_output.contiguous()
    )
    grad_weight = torch.empty((n_cols,), device=weight.device, dtype=weight.dtype)
    grad_weight_per_row = torch.empty((n_rows, n_cols), device=x.device, dtype=torch.float32)
    _fused_add_rmsnorm_bwd_kernel[(n_rows,)](
        x.contiguous(),
        residual.contiguous(),
        weight.contiguous(),
        grad_y_c,
        grad_updated_residual_c,
        grad_x,
        grad_residual,
        grad_weight_per_row,
        n_cols,
        eps,
        BLOCK_SIZE=triton.next_power_of_2(n_cols),
        num_warps=_NUM_WARPS,
        enable_fp_fusion=False,
    )
    # Launch on the same stream: the merge observes completed row contributions.
    _fused_add_rmsnorm_bwd_weight_kernel[(triton.cdiv(n_cols, _WEIGHT_BLOCK_SIZE),)](
        grad_weight_per_row,
        grad_weight,
        n_rows,
        n_cols,
        BLOCK_SIZE=_WEIGHT_BLOCK_SIZE,
        num_warps=_NUM_WARPS,
        enable_fp_fusion=False,
    )
    return grad_x, grad_residual, grad_weight


class _FusedAddRMSNormTritonFunction(torch.autograd.Function):
    """Connect the two outputs and three input gradients to PyTorch autograd."""

    @staticmethod
    def forward(ctx, x, residual, weight, eps):
        x_c = x.contiguous()
        residual_c = residual.contiguous()
        weight_c = weight.contiguous()
        outputs = _launch_fused_add_rmsnorm_fwd(x_c, residual_c, weight_c, eps=eps)
        ctx.save_for_backward(x_c, residual_c, weight_c)
        ctx.eps = eps
        # Backward explicitly handles an output that was not used by the loss.
        ctx.set_materialize_grads(False)
        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y, grad_updated_residual_output):
        if grad_y is None and grad_updated_residual_output is None:
            return None, None, None, None
        x, residual, weight = ctx.saved_tensors
        gradients = _launch_fused_add_rmsnorm_bwd(
            x, residual, weight, grad_y, grad_updated_residual_output, eps=ctx.eps
        )
        grad_x, grad_residual, grad_weight = (
            gradient if needed else None
            for gradient, needed in zip(gradients, ctx.needs_input_grad[:3], strict=True)
        )
        return grad_x, grad_residual, grad_weight, None


class TritonFusedAddRMSNormOp:
    """FP32-output prototype with first-order autograd on CUDA/ROCm devices.

    x/residual have identical shape [..., D], and weight has shape [D]. All
    inputs share a device and may independently use FP16, BF16, or FP32.
    Both outputs retain the input shape and use FP32; each input gradient uses
    that input's dtype. Noncontiguous tensors are copied to contiguous buffers.
    Nemotron's final output/residual rounding contract is still pending.
    """

    op_class = "norm"

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
        _validate_inputs(x, residual, weight, eps)
        return _FusedAddRMSNormTritonFunction.apply(x, residual, weight, eps)
