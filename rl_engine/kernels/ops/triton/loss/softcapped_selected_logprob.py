# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Triton forward/backward cores for Gemma's softcapped selected logprob.

Default forward uses one program per row. It reduces vocabulary tiles of 1024 elements in
ascending order, accumulating in FP32 with four warps and no FP contraction.
An experimental parallel forward writes the same tile sums to FP32 scratch,
then merges them in the same ascending order in a second kernel.
The schedule does not depend on batch size or the number of SMs/CUs. The
launchers prepare contiguous inputs; no full softcapped/probability buffer is
written to global memory. Forward saves one FP32 log_sum_exp per row so backward
can reuse it without repeating the reduction. Backward uses one program per
row/vocabulary tile, with disjoint gradient stores. The autograd wrapper saves the
inputs and these statistics. The registry exposes this as softcapped_selected_logprob.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Keep the accepted device types aligned with final_logit_softcap.
_SUPPORTED_DEVICES = ("cuda", "hip", "xpu", "musa")

_BLOCK_V = 1024


@triton.jit
def _softcapped_selected_logprob_fwd_kernel(
    logits_ptr,
    token_ids_ptr,
    selected_logprob_ptr,
    log_sum_exp_ptr,
    vocab_size: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # One program handles one row; widen before multiplying the row offset.
    row = tl.program_id(0).to(tl.int64)
    row_start = row * vocab_size
    cols = tl.arange(0, BLOCK_V)

    # 1. Accumulate sum(exp(softcap(logits))) in a fixed vocabulary order.
    sum_exp = tl.zeros((), dtype=tl.float32)
    for start in range(0, vocab_size, BLOCK_V):
        vocab_offsets = start + cols
        mask = vocab_offsets < vocab_size
        logits = tl.load(logits_ptr + row_start + vocab_offsets, mask=mask, other=0.0).to(tl.float32)  # noqa: E501 # fmt: skip

        scaled_logits = tl.div_rn(logits, 30.0)
        softcapped = 30.0 * libdevice.tanh(scaled_logits)
        exp_softcapped = tl.exp(softcapped)
        # Padding must contribute zero, not exp(softcap(0)) = 1.
        exp_softcapped = tl.where(mask, exp_softcapped, 0.0)
        tile_sum_exp = tl.sum(exp_softcapped, axis=0)
        sum_exp = sum_exp + tile_sum_exp

    # With softcap fixed at 30, no max-shift is needed to avoid overflow for
    # the Gemma vocabulary. This assumption must be revisited for other caps.
    log_sum_exp = tl.log(sum_exp)

    # 2. Load only the selected score and apply the same softcap arithmetic.
    token_id = tl.load(token_ids_ptr + row).to(tl.int64)
    selected_logit = tl.load(
        logits_ptr + row_start + token_id,
        mask=(token_id >= 0) & (token_id < vocab_size),
        other=float("nan"),
    ).to(tl.float32)
    selected_softcapped = 30.0 * libdevice.tanh(tl.div_rn(selected_logit, 30.0))

    # 3. Store one FP32 log-probability for this row.
    selected_logprob = selected_softcapped - log_sum_exp
    tl.store(selected_logprob_ptr + row, selected_logprob)
    tl.store(log_sum_exp_ptr + row, log_sum_exp)


@triton.jit
def _softcapped_selected_logprob_fwd_partial_kernel(
    logits_ptr,
    partial_sum_exp_ptr,
    vocab_size: tl.constexpr,
    n_tiles: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    row_start = row * vocab_size
    cols = tl.arange(0, BLOCK_V)
    vocab_offsets = tile * BLOCK_V + cols
    mask = vocab_offsets < vocab_size

    # Keep the original 1024-element tile arithmetic and reduction order.
    logits = tl.load(logits_ptr + row_start + vocab_offsets, mask=mask, other=0.0).to(tl.float32)  # noqa: E501 # fmt: skip
    scaled_logits = tl.div_rn(logits, 30.0)
    softcapped = 30.0 * libdevice.tanh(scaled_logits)
    exp_softcapped = tl.exp(softcapped)
    exp_softcapped = tl.where(mask, exp_softcapped, 0.0)
    tile_sum_exp = tl.sum(exp_softcapped, axis=0)

    # Each program owns exactly one entry in the [M, n_tiles] scratch array.
    tl.store(partial_sum_exp_ptr + row * n_tiles + tile, tile_sum_exp)


@triton.jit
def _softcapped_selected_logprob_fwd_merge_kernel(
    logits_ptr,
    token_ids_ptr,
    partial_sum_exp_ptr,
    selected_logprob_ptr,
    log_sum_exp_ptr,
    vocab_size: tl.constexpr,
    n_tiles: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_start = row * vocab_size
    partial_row_start = row * n_tiles

    # Preserve the original FP32 left-to-right accumulation, including the
    # initial zero. A tree reduction over partial sums would change rounding.
    sum_exp = tl.zeros((), dtype=tl.float32)
    for tile in range(0, n_tiles):
        tile_sum_exp = tl.load(partial_sum_exp_ptr + partial_row_start + tile)
        sum_exp = sum_exp + tile_sum_exp
    log_sum_exp = tl.log(sum_exp)

    token_id = tl.load(token_ids_ptr + row).to(tl.int64)
    selected_logit = tl.load(
        logits_ptr + row_start + token_id,
        mask=(token_id >= 0) & (token_id < vocab_size),
        other=float("nan"),
    ).to(tl.float32)
    selected_softcapped = 30.0 * libdevice.tanh(tl.div_rn(selected_logit, 30.0))
    selected_logprob = selected_softcapped - log_sum_exp
    tl.store(selected_logprob_ptr + row, selected_logprob)
    tl.store(log_sum_exp_ptr + row, log_sum_exp)


@triton.jit
def _softcapped_selected_logprob_bwd_kernel(
    logits_ptr,
    token_ids_ptr,
    grad_selected_logprob_ptr,
    log_sum_exp_ptr,
    grad_logits_ptr,
    vocab_size: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    row_start = row * vocab_size
    cols = tl.arange(0, BLOCK_V)
    vocab_offsets = tile * BLOCK_V + cols
    mask = vocab_offsets < vocab_size

    # 1. Reuse the FP32 log_sum_exp saved by forward for this row.
    log_sum_exp = tl.load(log_sum_exp_ptr + row)
    token_id = tl.load(token_ids_ptr + row).to(tl.int64)
    grad_selected_logprob = tl.load(grad_selected_logprob_ptr + row).to(tl.float32)

    # 2. Each program computes one tile using the shared, read-only row statistics.
    logits = tl.load(logits_ptr + row_start + vocab_offsets, mask=mask, other=0.0).to(tl.float32)  # noqa: E501 # fmt: skip

    scaled_logits = tl.div_rn(logits, 30.0)
    tanh_scaled_logits = libdevice.tanh(scaled_logits)
    softcapped = 30.0 * tanh_scaled_logits
    probability = tl.exp(softcapped - log_sum_exp)

    # 3. Chain through selected logprob: upstream * (selected - p).
    # 1.0 at the selected token position; 0.0 at all other positions.
    is_selected = tl.where(vocab_offsets == token_id, 1.0, 0.0)
    grad_softcapped = grad_selected_logprob * (is_selected - probability)

    # 4. Chain through softcap; each gradient element has exactly one writer.
    softcap_derivative = 1.0 - tanh_scaled_logits * tanh_scaled_logits
    grad_logits = grad_softcapped * softcap_derivative
    grad_logits = tl.where((token_id >= 0) & (token_id < vocab_size), grad_logits, float("nan"))
    tl.store(
        grad_logits_ptr + row_start + vocab_offsets,
        grad_logits.to(grad_logits_ptr.dtype.element_ty),
        mask=mask,
    )


def _validate_inputs(logits: torch.Tensor, token_ids: torch.Tensor) -> None:
    if logits.device.type not in _SUPPORTED_DEVICES:
        raise RuntimeError(
            "The Triton core requires a GPU tensor; "
            f"supported device types are {_SUPPORTED_DEVICES}, got '{logits.device.type}'."
        )
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("logits must have shape [M, V] with V > 0.")
    if logits.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"logits must have dtype {_SUPPORTED_DTYPES}, got {logits.dtype}.")
    if token_ids.shape != logits.shape[:1]:
        raise ValueError("token_ids must have shape [M], matching the logits rows.")
    if token_ids.dtype != torch.int64 or token_ids.device != logits.device:
        raise TypeError("token_ids must have dtype int64 and be on the logits device.")


def _validate_backward_inputs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    grad_selected_logprob: torch.Tensor,
    log_sum_exp: torch.Tensor,
) -> None:
    _validate_inputs(logits, token_ids)
    if grad_selected_logprob.shape != logits.shape[:1]:
        raise ValueError("grad_selected_logprob must have shape [M].")
    if grad_selected_logprob.device != logits.device:
        raise ValueError("grad_selected_logprob must be on the logits device.")
    if grad_selected_logprob.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"grad_selected_logprob must have dtype {_SUPPORTED_DTYPES}, "
            f"got {grad_selected_logprob.dtype}."
        )
    if log_sum_exp.shape != logits.shape[:1]:
        raise ValueError("log_sum_exp must have shape [M].")
    if log_sum_exp.device != logits.device:
        raise ValueError("log_sum_exp must be on the logits device.")
    if log_sum_exp.dtype != torch.float32:
        raise TypeError("log_sum_exp must have dtype float32.")


def _launch_softcapped_selected_logprob_fwd(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 selected_logprob and log_sum_exp, both [M], without autograd.

    Valid token IDs in [0, V) are a caller precondition. The kernel masks
    invalid IDs to NaN to avoid out-of-bounds reads, without a host/GPU sync.
    """
    _validate_inputs(logits, token_ids)

    n_rows, vocab_size = logits.shape
    logits_c = logits.contiguous()
    token_ids_c = token_ids.contiguous()
    output = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    log_sum_exp = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    if n_rows == 0:
        return output, log_sum_exp

    _softcapped_selected_logprob_fwd_kernel[(n_rows,)](
        logits_c,
        token_ids_c,
        output,
        log_sum_exp,
        vocab_size,
        BLOCK_V=_BLOCK_V,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, log_sum_exp


def _launch_softcapped_selected_logprob_fwd_parallel(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Experimental two-kernel forward with FP32 [M, ceil(V / 1024)] scratch.

    Return the same FP32 [M] output/statistics pair as the row-loop version.
    Allocation, partial reduction and ordered merge all belong to this call.
    """
    _validate_inputs(logits, token_ids)

    n_rows, vocab_size = logits.shape
    logits_c = logits.contiguous()
    token_ids_c = token_ids.contiguous()
    output = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    log_sum_exp = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    if n_rows == 0:
        return output, log_sum_exp

    n_tiles = triton.cdiv(vocab_size, _BLOCK_V)
    partial_sum_exp = torch.empty((n_rows, n_tiles), device=logits.device, dtype=torch.float32)
    _softcapped_selected_logprob_fwd_partial_kernel[(n_rows, n_tiles)](
        logits_c,
        partial_sum_exp,
        vocab_size,
        n_tiles,
        BLOCK_V=_BLOCK_V,
        num_warps=4,
        enable_fp_fusion=False,
    )
    # Both kernels launch on the current stream, so the merge reads completed
    # partial sums without an explicit host synchronization.
    _softcapped_selected_logprob_fwd_merge_kernel[(n_rows,)](
        logits_c,
        token_ids_c,
        partial_sum_exp,
        output,
        log_sum_exp,
        vocab_size,
        n_tiles,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, log_sum_exp


def _launch_softcapped_selected_logprob_bwd(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    grad_selected_logprob: torch.Tensor,
    log_sum_exp: torch.Tensor,
) -> torch.Tensor:
    """Return input-dtype grad_logits; grad_selected_logprob has shape [M].

    log_sum_exp is the FP32 [M] statistic returned by forward for these inputs.
    The autograd wrapper supplies it together with the inputs and upstream gradient.
    As in forward, invalid token IDs produce NaN; valid IDs are a precondition.
    """
    _validate_backward_inputs(logits, token_ids, grad_selected_logprob, log_sum_exp)

    n_rows, vocab_size = logits.shape
    logits_c = logits.contiguous()
    token_ids_c = token_ids.contiguous()
    grad_selected_logprob_c = grad_selected_logprob.contiguous()
    log_sum_exp_c = log_sum_exp.contiguous()
    grad_logits = torch.empty_like(logits_c)
    if n_rows == 0:
        return grad_logits

    grid = (n_rows, triton.cdiv(vocab_size, _BLOCK_V))
    _softcapped_selected_logprob_bwd_kernel[grid](
        logits_c,
        token_ids_c,
        grad_selected_logprob_c,
        log_sum_exp_c,
        grad_logits,
        vocab_size,
        BLOCK_V=_BLOCK_V,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return grad_logits


class _SoftcappedSelectedLogprobTritonFunction(torch.autograd.Function):
    """Connect the Triton forward/backward cores to PyTorch autograd."""

    @staticmethod
    def forward(
        ctx, logits: torch.Tensor, token_ids: torch.Tensor, forward_impl: str
    ) -> torch.Tensor:
        logits_c = logits.contiguous()
        token_ids_c = token_ids.contiguous()
        launch_forward = (
            _launch_softcapped_selected_logprob_fwd_parallel
            if forward_impl == "parallel"
            else _launch_softcapped_selected_logprob_fwd
        )
        selected_logprob, log_sum_exp = launch_forward(logits_c, token_ids_c)
        ctx.save_for_backward(logits_c, token_ids_c, log_sum_exp)
        return selected_logprob

    @staticmethod
    def backward(ctx, grad_selected_logprob: torch.Tensor):
        logits, token_ids, log_sum_exp = ctx.saved_tensors
        grad_logits = None
        if ctx.needs_input_grad[0]:
            grad_logits = _launch_softcapped_selected_logprob_bwd(
                logits, token_ids, grad_selected_logprob, log_sum_exp
            )
        # Integer token IDs and the forward implementation selector have no gradient.
        return grad_logits, None, None


class TritonSoftcappedSelectedLogprobOp:
    """Selected logprob after FP32 softcap, with first-order autograd.

    logits [M, V] must use a supported floating dtype on an accepted GPU; token_ids
    [M] must be int64 on the same device with values in [0, V). Output is FP32
    [M], and gradients use the logits dtype. Softcap is fixed at 30.0.
    forward_impl="row" keeps the original loop; "parallel" explicitly opts in
    to the experimental two-kernel forward. Both reuse the same backward.
    """

    op_class = "logprob"

    def __init__(self, *, forward_impl: str = "row"):
        if forward_impl not in ("row", "parallel"):
            raise ValueError("forward_impl must be 'row' or 'parallel'.")
        self.forward_impl = forward_impl

    def __call__(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(logits, token_ids)

    def forward(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return _SoftcappedSelectedLogprobTritonFunction.apply(logits, token_ids, self.forward_impl)
