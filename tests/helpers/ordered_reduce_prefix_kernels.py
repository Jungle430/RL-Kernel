# SPDX-License-Identifier: Apache-2.0
"""Test-only launches with physical row stride independent of effective length."""

import triton
import triton.language as tl

from rl_engine.backends.shared.triton.reduction.ordered_reduce import ordered_reduce_partial


@triton.jit
def prefix_vector_probe(
    input_ptr,
    output_ptr,
    row_stride,
    LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    REDUCE_FN: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    """Reduce one effective prefix per program, leaving physical padding unread."""
    row = tl.program_id(0).to(tl.int64)
    total = REDUCE_FN(
        input_ptr + row * row_stride,
        LENGTH,
        COMPUTE_DTYPE=tl.float32,
        BLOCK_SIZE=BLOCK_SIZE,
        PIPELINE_STAGES=PIPELINE_STAGES,
    )
    tl.store(output_ptr + row, total)


@triton.jit
def prefix_partials_probe(
    input_ptr,
    partials_ptr,
    row_stride,
    LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COMBINE_FN: tl.constexpr,
    IDENTITY: tl.constexpr,
    REVERSE_TILES: tl.constexpr,
):
    """Change producer-to-tile assignment without changing logical tile slots."""
    row = tl.program_id(0).to(tl.int64)
    n_tiles: tl.constexpr = (LENGTH + BLOCK_SIZE - 1) // BLOCK_SIZE
    tile = tl.program_id(1).to(tl.int64)
    if REVERSE_TILES:
        tile = n_tiles - 1 - tile
    partial = ordered_reduce_partial(
        input_ptr + row * row_stride,
        LENGTH,
        tile,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_GROUP_SIZE=8,
        SECTION_COUNT=4,
        COMBINE_FN=COMBINE_FN,
        IDENTITY=IDENTITY,
        COMPUTE_DTYPE=tl.float32,
    )
    tl.store(partials_ptr + row * n_tiles + tile, partial)
