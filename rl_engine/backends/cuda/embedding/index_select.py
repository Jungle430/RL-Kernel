# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H100 NVL index selection with C++ allocation and ordered CUDA backward."""

from __future__ import annotations

from functools import lru_cache
from math import prod

import torch
from torch.autograd.function import once_differentiable

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.ops.autograd.backward_runtime import record_backward

KERNEL_ID = "rl_engine._C.index_select_h100_forward"
BACKWARD_KERNEL_ID = "rl_engine._C.index_select_h100_backward"
BACKWARD_IMPL = "stable_segment_sequential_fp32_cast_once"
_MAX_ELEMENTS = 2**31 - 1
_MAX_WORKSPACE_BYTES = 128 * 1024 * 1024


def index_select_h100_available() -> bool:
    """No compilation or CUDA initialization at module import/registry load."""
    return bool(
        _EXT_AVAILABLE
        and hasattr(_C, "index_select_h100_forward")
        and hasattr(_C, "index_select_h100_backward")
    )


@lru_cache(maxsize=None)
def _qualified_device(device_index: int) -> bool:
    """Cache immutable hardware properties by the actual tensor device ordinal."""
    properties = torch.cuda.get_device_properties(device_index)
    return (properties.major, properties.minor) == (9, 0) and properties.name == "NVIDIA H100 NVL"


def _stream_is_capturing(device_index: int) -> bool:
    """Query the input device's current stream and restore the caller's device."""
    with torch.cuda.device(device_index):
        return torch.cuda.is_current_stream_capturing()


def _supported_shape(x: torch.Tensor, index: torch.Tensor, dim: int) -> bool:
    """Bound native integer addressing and workspace using host metadata only.

    Invalid metadata is left to the selected backend's ordinary validation;
    this predicate never synchronizes or inspects index contents.
    """
    if not 1 <= x.ndim <= 32 or not -x.ndim <= dim < x.ndim or index.ndim != 1:
        return False
    # Empty tensors can have huge intermediate prefix products despite zero
    # logical elements. Keep those layouts on the general implementation.
    if x.numel() == 0:
        return False
    axis = dim % x.ndim
    outer, inner = prod(x.shape[:axis]), prod(x.shape[axis + 1 :])
    if max((*x.shape, outer, inner, x.numel(), outer * index.numel() * inner)) >= _MAX_ELEMENTS:
        return False
    if index.numel() >= _MAX_ELEMENTS:
        return False
    # At least 32 integer position slots plus a count per input-axis entry.
    return (x.shape[axis] * 33 + 1 + max(1, index.numel() // 5)) * 4 <= _MAX_WORKSPACE_BYTES


class _H100IndexSelectFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, index, dim):
        output = _C.index_select_h100_forward(x, index, dim)
        ctx.save_for_backward(index)
        ctx.input_shape = x.shape
        ctx.dim = dim
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        result = _C.index_select_h100_backward(grad, index, ctx.input_shape, ctx.dim)
        record_backward(
            "index_select", kernel_id=BACKWARD_KERNEL_ID, impl=BACKWARD_IMPL, family="cuda"
        )
        return result, None, None


class H100IndexSelectOp:
    """Strict H100 NVL backend for the public fixed-order index_select contract.

    The native entry validates shape/dtype/device, allocates output and bounded
    integer workspace, and launches the kernels on the input device's current
    stream. Forward copies storage bits. Backward adds occurrences in increasing
    original position from FP32 +0, with one final cast. No floating-point
    atomics, partial sums, or cross-coordinate reductions are used.

    First-order autograd only. The qualified native subset has rank <= 32,
    int32-sized logical tensors and at most 128 MiB of integer workspace.
    Use CudaIndexSelectOp for automatic routing to the shared implementation.
    """

    op_class = "reduction"
    is_batch_invariant = True
    kernel_id = KERNEL_ID
    backward_impl = BACKWARD_IMPL

    def __init__(self):
        if not index_select_h100_available():
            raise RuntimeError("rebuild rl_engine._C without fast math for index_select_h100")

    def __call__(self, x: torch.Tensor, index: torch.Tensor, *, dim: int) -> torch.Tensor:
        return self.forward(x, index, dim=dim)

    def forward(self, x: torch.Tensor, index: torch.Tensor, *, dim: int) -> torch.Tensor:
        return _H100IndexSelectFunction.apply(x, index, dim)


class CudaIndexSelectOp:
    """Legacy-registry dispatcher, selecting hardware from each actual input.

    Registry construction and get_op(device=...) cannot know which tensor will
    later be passed. Keep the hardware guard here, so a noncurrent-device input
    and a heterogeneous process both select correctly without changing the
    process's current device. H100 eager calls use the native entry to reduce
    host submission cost. Graph capture retains Triton: measured native device
    replay is slower on most qualified workloads. Other CUDA devices/shapes
    also use Triton. The strict native backend remains explicitly selectable.
    """

    op_class = "reduction"
    is_batch_invariant = True

    def __init__(self):
        from rl_engine.backends.shared.triton.embedding.index_select import TritonIndexSelectOp

        self.h100 = H100IndexSelectOp()
        self.shared = TritonIndexSelectOp()

    def __call__(self, x: torch.Tensor, index: torch.Tensor, *, dim: int) -> torch.Tensor:
        return self.forward(x, index, dim=dim)

    def forward(self, x: torch.Tensor, index: torch.Tensor, *, dim: int) -> torch.Tensor:
        if (
            x.device.type == "cuda"
            and torch.version.hip is None
            and _qualified_device(x.device.index)
            and _supported_shape(x, index, dim)
            and not _stream_is_capturing(x.device.index)
        ):
            return self.h100(x, index, dim=dim)
        return self.shared(x, index, dim=dim)
