# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared index selection: ordered FP32 reference and independent accuracy oracle."""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_INDEX_DTYPES = (torch.int32, torch.int64)


def _validate_inputs(x: torch.Tensor, index: torch.Tensor, dim: int) -> int:
    """Validate tensors and bounds, returning the normalized selection axis.

    CUDA range errors are asynchronous device assertions, like index_select.
    The CPU reference raises IndexError synchronously.
    """
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    if not -x.ndim <= dim < x.ndim:
        raise IndexError(f"dim must be in [{-x.ndim}, {x.ndim}), got {dim}")
    dim = dim % x.ndim
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"x dtype must be one of {_SUPPORTED_DTYPES}, got {x.dtype}")
    if index.ndim != 1:
        raise ValueError("index must have shape [K]")
    if index.dtype not in _INDEX_DTYPES:
        raise TypeError(f"index dtype must be one of {_INDEX_DTYPES}, got {index.dtype}")
    if x.device != index.device:
        raise ValueError("x and index must be on the same device")
    if index.numel():
        valid = ((index >= 0) & (index < x.shape[dim])).all()
        message = f"index values must be in [0, {x.shape[dim]})"
        if x.is_cuda:
            torch._assert_async(valid, message)
        elif not bool(valid):
            raise IndexError(message)
    return dim


class _OrderedSelect(torch.autograd.Function):
    """Copy selected slices and explicitly implement the ordered input VJP."""

    @staticmethod
    def forward(ctx, x, index, dim):
        ctx.save_for_backward(index)
        ctx.x_shape = x.shape
        ctx.input_dtype = x.dtype
        ctx.dim = dim
        return x.index_select(dim, index)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        # Independent executable contract: visit output positions in increasing j.
        # Use CPU tensors to avoid CUDA atomics and one device launch per position.
        grad32 = grad.detach().to(device="cpu", dtype=torch.float32)
        result = torch.zeros(ctx.x_shape, dtype=torch.float32)
        for position, row in enumerate(index.cpu().tolist()):
            result.select(ctx.dim, row).add_(grad32.select(ctx.dim, position))
        return result.to(device=grad.device, dtype=ctx.input_dtype), None, None


class _AccuracySelect(torch.autograd.Function):
    """Use an independent FP64 scatter-add for numerical accuracy checks."""

    @staticmethod
    def forward(ctx, x, index, dim):
        ctx.save_for_backward(index)
        ctx.x_shape = x.shape
        ctx.input_dtype = x.dtype
        ctx.dim = dim
        return x.float().index_select(dim, index)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        # An independent CPU FP64 scatter-add checks accuracy, not bitwise order.
        result = torch.zeros(ctx.x_shape, dtype=torch.float64)
        result.index_add_(ctx.dim, index.cpu().long(), grad.detach().double().cpu())
        return result.to(device=grad.device, dtype=ctx.input_dtype), None, None


class NativeIndexSelectOp:
    """Select slices along dim with a fixed-order FP32 input VJP.

    x has rank >= 1 and FP32/FP16/BF16 dtype. index is a same-device int32/int64
    vector [K] shared by all remaining coordinates. dim is a required keyword;
    negative dimensions are normalized, while negative indices are rejected.
    Output replaces x.shape[dim] by K, is contiguous, and preserves copied bits.
    Repeated indices accumulate in increasing output position, independently at
    every other coordinate: start at FP32 +0, round each addition to FP32, then
    cast once to x.dtype. Unselected coordinates receive +0. Strided and empty
    tensors are supported, including a zero-length selected axis with empty index.
    This reference intentionally copies gradients to CPU; it is not a baseline
    for performance. Only first-order autograd is supported.
    """

    op_class = "reduction"

    def __call__(self, x: torch.Tensor, index: torch.Tensor, *, dim: int) -> torch.Tensor:
        return self.forward(x, index, dim=dim)

    def forward(self, x: torch.Tensor, index: torch.Tensor, *, dim: int) -> torch.Tensor:
        """Preserve input storage bits and use ordered FP32 sums in backward."""
        dim = _validate_inputs(x, index, dim)
        return _OrderedSelect.apply(x, index, dim)

    def forward_fp32(self, x: torch.Tensor, index: torch.Tensor, *, dim: int) -> torch.Tensor:
        """FP32 output with independent CPU FP64 VJP for accuracy validation."""
        dim = _validate_inputs(x, index, dim)
        return _AccuracySelect.apply(x, index, dim)
