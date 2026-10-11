# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Readable FP32 reference for softcapped selected logprob."""

import math

import torch
from torch import nn

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_softcap(softcap: float, vocab_size: int) -> None:
    """Check the scalar and the FP32 exp/sum range for a positive vocabulary width.

    Rely on type annotations. Require a finite positive FP32-normal cap,
    exp(-softcap) >= 2 * float32.tiny and V * exp(softcap) <= float32.max / 2.
    The factor-of-two margins keep exponentials and their sum away from limits.
    """
    limits = torch.finfo(torch.float32)
    if not math.isfinite(softcap) or softcap < limits.tiny:
        raise ValueError("softcap must be finite, positive and at least float32.tiny.")
    max_softcap = min(
        -math.log(2.0 * limits.tiny), math.log(limits.max / 2.0) - math.log(vocab_size)
    )
    if softcap > max_softcap:
        raise ValueError(
            f"softcap={softcap} with V={vocab_size} exceeds the FP32 exp/sum range "
            f"(maximum {max_softcap:.6g})."
        )


def _validate_inputs(logits: torch.Tensor, token_ids: torch.Tensor, softcap: float) -> None:
    """Validate tensors and the FP32 row-sum bound for their vocabulary width."""
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("logits must have shape [M, V] with V > 0.")
    _validate_softcap(softcap, logits.shape[1])
    if logits.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"logits must have dtype {_SUPPORTED_DTYPES}, got {logits.dtype}.")
    if token_ids.shape != logits.shape[:1]:
        raise ValueError("token_ids must have shape [M], matching the logits rows.")
    if token_ids.dtype != torch.int64 or token_ids.device != logits.device:
        raise TypeError("token_ids must have dtype int64 and be on the logits device.")


def softcapped_selected_logprob(
    logits: torch.Tensor, token_ids: torch.Tensor, softcap: float = 30.0
) -> torch.Tensor:
    """Return one FP32 log-probability per row; PyTorch supplies autograd.

    Inputs: logits [M, V] in FP16/BF16/FP32, and int64 token_ids [M] on the
    same device. V must be positive and each token ID must lie in [0, V).
    Output: selected_logprob [M]. softcap defaults to 30.0 for Gemma; callers
    may pass another positive scalar within the FP32 exp/sum range.

    This reference is exposed by NativeSoftcappedSelectedLogprobOp. Its
    reduction order is not the Triton kernel's fixed tiled reduction order.
    """
    _validate_inputs(logits, token_ids, softcap)
    softcap = float(softcap)

    # 1. Apply softcap in FP32; the default retains the original arithmetic.
    logits_f = logits.to(torch.float32)
    softcapped = softcap * torch.tanh(logits_f / softcap)

    # 2. Sum exponentials across each row, then take the natural logarithm.
    # Validation bounds exp(-softcap) and V * exp(softcap) for this FP32 path.
    exp_softcapped = torch.exp(softcapped)
    sum_exp = exp_softcapped.sum(dim=-1)
    log_sum_exp = torch.log(sum_exp)

    # 3. Select one softcapped score per row; [M, 1] becomes [M].
    selected_softcapped = softcapped.gather(dim=-1, index=token_ids[:, None]).squeeze(-1)

    # 4. log(exp(selected) / sum_exp) = selected - log(sum_exp).
    selected_logprob = selected_softcapped - log_sum_exp
    return selected_logprob


class NativeSoftcappedSelectedLogprobOp(nn.Module):
    """PyTorch softcapped selected logprob with FP32 output and native autograd.

    logits has shape [M, V] and token_ids has shape [M]. Valid token IDs lie in
    [0, V). Each forward call accepts softcap, defaulting to 30.0; it is a
    positive scalar without its own gradient. Forward validates its FP32
    exp/sum range for V. The Op does not store a softcap value between calls.
    The forward's native operations supply input gradients automatically.
    """

    op_class = "logprob"

    def forward(
        self, logits: torch.Tensor, token_ids: torch.Tensor, *, softcap: float = 30.0
    ) -> torch.Tensor:
        """Compute with this call's softcap; PyTorch records it in the native graph."""
        return softcapped_selected_logprob(logits, token_ids, softcap)
