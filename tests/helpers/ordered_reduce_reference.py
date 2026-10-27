# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Test-only collection of fixed-order partials and their merged total.

Compose the PyTorch stage references to inspect intermediate results in tests.
The production reference module mirrors Triton's helpers; this collector is not
part of either public reduction package and defines no additional arithmetic.
"""

from dataclasses import dataclass

import torch

from rl_engine.reference.reduction.ordered_reduce import (
    _Combine,
    ordered_reduce_merge,
    ordered_reduce_partial,
)


@dataclass(frozen=True)
class OrderedReduceResult:
    """Test-only diagnostic trace of the partial and merge stages."""

    partials: torch.Tensor
    """compute_dtype [..., ceil(length / block_size)] totals in ascending tile order."""

    total: torch.Tensor
    """compute_dtype [...] left fold of partials, starting with initial once."""


def ordered_reduce(
    values: torch.Tensor,
    *,
    block_size: int,
    max_group_size: int,
    combine_fn: _Combine,
    identity: bool | int | float,
    initial: bool | int | float | torch.Tensor,
    compute_dtype: torch.dtype,
) -> OrderedReduceResult:
    """Collect inspectable partials and total for operator-stage diagnostics.

    This PyTorch-only collector composes ordered_reduce_partial and
    ordered_reduce_merge; it does not define another tree. Use the matching
    stage helpers or sum/product with from_memory=True when only a total is needed.
    Leading dimensions batch independent vectors; the last dimension must be
    nonempty. All arithmetic parameters are required, with the same meanings
    as in the stage helpers. initial never enters the unseeded partials.
    """
    if values.ndim == 0 or values.shape[-1] == 0:
        raise ValueError("ordered_reduce requires values with a nonempty last dimension")
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError("block_size must be a positive power of two")
    length = values.shape[-1]
    n_tiles = (length + block_size - 1) // block_size
    tile_partials: list[torch.Tensor] = []
    for tile in range(n_tiles):
        tile_partials.append(
            ordered_reduce_partial(
                values,
                tile,
                block_size=block_size,
                max_group_size=max_group_size,
                combine_fn=combine_fn,
                identity=identity,
                compute_dtype=compute_dtype,
            )
        )
    partials = torch.stack(tile_partials, dim=-1)
    total = ordered_reduce_merge(
        partials,
        initial,
        combine_fn=combine_fn,
        merge_unroll=1,
        compute_dtype=compute_dtype,
    )
    return OrderedReduceResult(partials=partials, total=total)
