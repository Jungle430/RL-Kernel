# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared index selection with an ordered input gradient and integer-only grouping."""

from __future__ import annotations

from math import prod

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable

from rl_engine.ops.autograd.backward_runtime import record_backward

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_INDEX_DTYPES = (torch.int32, torch.int64)
_MIN_CUDA_SM = 80
_BLOCK_SIZE = 256
_MAX_GRID_Y = 65535
_SMALL_SORT_LIMIT = 2048
_MAX_BUCKET_BYTES = 128 * 1024 * 1024

KERNEL_ID = __name__ + "._index_select_fwd_kernel"
BACKWARD_KERNEL_ID = "stable_index_sort+" + __name__ + "._index_select_bwd_kernel"
BACKWARD_IMPL = "stable_segment_sequential_fp32_cast_once"


@triton.jit
def _strided_offset(flat_index, SHAPE: tl.constexpr, STRIDES: tl.constexpr):
    """Map a flattened logical coordinate to its strided element offset.

    Used separately for the dimensions before and after the selected axis.
    An empty shape contributes zero, covering selection along the first or
    last dimension without requiring an input copy.
    """
    offset = tl.full(flat_index.shape, 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        offset += (flat_index % SHAPE[axis]) * STRIDES[axis]
        flat_index = flat_index // SHAPE[axis]
    return offset


@triton.jit
def _index_select_fwd_kernel(
    x_ptr,
    index_ptr,
    y_ptr,
    n_selected: tl.constexpr,
    axis_size: tl.constexpr,
    outer_size: tl.constexpr,
    inner_size: tl.constexpr,
    OUTER_SHAPE: tl.constexpr,
    INNER_SHAPE: tl.constexpr,
    X_OUTER_STRIDES: tl.constexpr,
    X_INNER_STRIDES: tl.constexpr,
    x_stride_axis,
    index_stride,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Copy a tile from an indexed slice, retaining the input's storage bits.

    Logically view x as [outer, axis, inner] and y as [outer, K, inner].
    Grid axes are (blocks of outer * K rows, capped inner tiles). Each program strides over
    any remaining tiles beyond CUDA's grid-y limit. This view only
    describes indexing: x is read through its original strides, while y is
    contiguous. No floating-point arithmetic or input reshape is performed.
    """
    output_row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    valid_rows = output_row < outer_size * n_selected
    outer, position = output_row // n_selected, output_row % n_selected
    outer_offset = _strided_offset(outer, OUTER_SHAPE, X_OUTER_STRIDES)
    input_row = tl.load(index_ptr + position * index_stride, valid_rows, other=0).to(tl.int64)
    valid_index = (input_row >= 0) & (input_row < axis_size)
    # The launcher sets debug=True so this check is enabled independently of
    # TRITON_DEBUG. Invalid indices are also excluded from memory accesses.
    tl.device_assert(~valid_rows | valid_index, "index values must be within the selected axis")

    first_tile = tl.program_id(1)
    n_tiles = tl.cdiv(inner_size, BLOCK_SIZE)
    tile_stride = tl.num_programs(1)
    for tile in range(first_tile, n_tiles, tile_stride):
        cols = tile.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = valid_rows[:, None] & (cols[None, :] < inner_size)
        inner_offsets = _strided_offset(cols, INNER_SHAPE, X_INNER_STRIDES)
        value = tl.load(
            x_ptr
            + outer_offset[:, None]
            + input_row[:, None] * x_stride_axis
            + inner_offsets[None, :],
            mask=mask & valid_index[:, None],
            other=0,
        )
        tl.store(y_ptr + output_row[:, None] * inner_size + cols[None, :], value, mask=mask)


@triton.jit
def _index_select_fwd_contiguous_kernel(
    x_ptr,
    index_ptr,
    y_ptr,
    axis_size: tl.constexpr,
    n_selected: tl.constexpr,
    inner_size: tl.constexpr,
    index_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy a contiguous input using a 2-D (outer, output tiles) grid.

    Flatten only the output's selected and inner dimensions. Consecutive lanes
    therefore write consecutive elements, including across narrow slice edges.
    The input index vector may still be strided. Bounds checks stay in the copy
    kernel, and the capped second grid axis strides over longer outputs.
    """
    outer = tl.program_id(0).to(tl.int64)
    first_tile = tl.program_id(1)
    n_tiles = tl.cdiv(n_selected * inner_size, BLOCK_SIZE)
    tile_stride = tl.num_programs(1)
    for tile in range(first_tile, n_tiles, tile_stride):
        offsets = tile.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < n_selected * inner_size
        position, column = offsets // inner_size, offsets % inner_size
        source = tl.load(index_ptr + position * index_stride, valid, other=0).to(tl.int64)
        valid_index = (source >= 0) & (source < axis_size)
        tl.device_assert(~valid | valid_index, "index values must be within the selected axis")
        value = tl.load(
            x_ptr + (outer * axis_size + source) * inner_size + column,
            valid & valid_index,
            other=0,
        )
        tl.store(y_ptr + outer * n_selected * inner_size + offsets, value, valid)


@triton.jit
def _index_select_check_bounds_kernel(
    index_ptr,
    n_selected,
    axis_size: tl.constexpr,
    index_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Check indices when empty outer/inner dimensions leave no copy work."""
    positions = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = positions < n_selected
    values = tl.load(index_ptr + positions * index_stride, mask, other=0)
    valid = (values >= 0) & (values < axis_size)
    tl.device_assert(~mask | valid, "index values must be within the selected axis")


@triton.jit
def _index_select_bwd_small_kernel(
    grad_y_ptr,
    index_ptr,
    grad_x_ptr,
    n_selected: tl.constexpr,
    axis_size: tl.constexpr,
    inner_size: tl.constexpr,
    OUTER_SHAPE: tl.constexpr,
    INNER_SHAPE: tl.constexpr,
    GRAD_OUTER_STRIDES: tl.constexpr,
    GRAD_INNER_STRIDES: tl.constexpr,
    grad_y_stride_axis,
    index_stride,
    INDEX_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Gather a small selection's gradient directly, including unselected zeros.

    A program owns one source slice. Integer minima find its occurrences in
    increasing original position; long groups sort those integer positions once
    before prefetching. Only gradient additions use FP32. This avoids separate
    zero-fill and sort launches for bounded small workloads.
    Searching every source is quadratic in the index dimensions, so the
    launcher restricts this path instead of applying it to large selections.
    """
    row = tl.program_id(0).to(tl.int64)
    outer, input_row = row // axis_size, row % axis_size
    positions = tl.arange(0, INDEX_BLOCK)
    indices = tl.load(index_ptr + positions * index_stride, positions < n_selected, other=-1)
    candidates = tl.where((positions < n_selected) & (indices == input_row), positions, n_selected)
    first = tl.min(candidates, axis=0)
    count = tl.sum((candidates < n_selected).to(tl.int32), axis=0)
    ordered_positions = candidates
    # Prune sorting/prefetch at compile time for vectors smaller than a window.
    if n_selected > 8:
        if count > 8:
            ordered_positions = tl.sort(candidates, descending=False)
    outer_offset = _strided_offset(outer, OUTER_SHAPE, GRAD_OUTER_STRIDES)
    first_tile = tl.program_id(1)
    n_tiles = tl.cdiv(inner_size, BLOCK_SIZE)
    tile_stride = tl.num_programs(1)
    for tile in range(first_tile, n_tiles, tile_stride):
        cols = tile.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        inner_offsets = _strided_offset(cols, INNER_SHAPE, GRAD_INNER_STRIDES)
        total = tl.full((BLOCK_SIZE,), 0, tl.float32)
        if n_selected <= 8 or count <= 8:
            cursor = first
            while cursor < n_selected:
                value = tl.load(
                    grad_y_ptr
                    + outer_offset
                    + cursor.to(tl.int64) * grad_y_stride_axis
                    + inner_offsets,
                    cols < inner_size,
                    other=0,
                ).to(tl.float32)
                total = total + value
                cursor = tl.min(tl.where(candidates > cursor, candidates, n_selected), axis=0)
        else:
            # One source owns a long chain. Compact its integer positions once,
            # then prefetch eight upstreams, without reassociating the additions.
            for window in range(tl.cdiv(count, 8)):
                offsets = window * 8 + tl.arange(0, 8)
                selected_positions = tl.gather(
                    ordered_positions, tl.minimum(offsets, INDEX_BLOCK - 1), axis=0
                )
                values = tl.load(
                    grad_y_ptr
                    + outer_offset
                    + selected_positions[:, None].to(tl.int64) * grad_y_stride_axis
                    + inner_offsets[None, :],
                    (offsets[:, None] < count) & (cols[None, :] < inner_size),
                    other=0,
                ).to(tl.float32)
                for offset in tl.static_range(8):
                    value = tl.reshape(
                        tl.gather(values, tl.full((1, BLOCK_SIZE), offset, tl.int32), axis=0),
                        (BLOCK_SIZE,),
                    )
                    total = tl.where(window * 8 + offset < count, total + value, total)
        tl.store(
            grad_x_ptr + row * inner_size + cols,
            total.to(grad_x_ptr.dtype.element_ty),
            cols < inner_size,
        )


@triton.jit
def _index_select_sort_kernel(
    index_ptr,
    sorted_index_ptr,
    order_ptr,
    n_selected: tl.constexpr,
    index_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sort a small index vector by (input index, original output position).

    The launcher proves that every packed key fits in positive int32. Multiplying
    the input index by BLOCK_SIZE leaves room for the original position, making
    real keys unique. Thus even an unstable integer sort preserves occurrence
    order among duplicate input indices. Padding sorts after every real key.
    """
    positions = tl.arange(0, BLOCK_SIZE)
    mask = positions < n_selected
    indices = tl.load(index_ptr + positions * index_stride, mask, other=0).to(tl.int32)
    keys = indices * BLOCK_SIZE + positions
    keys = tl.where(mask, keys, 2147483647)
    sorted_keys = tl.sort(keys, descending=False)
    tl.store(sorted_index_ptr + positions, sorted_keys // BLOCK_SIZE, mask)
    tl.store(order_ptr + positions, sorted_keys % BLOCK_SIZE, mask)


@triton.jit
def _index_select_bwd_kernel(
    sorted_index_ptr,
    order_ptr,
    grad_y_ptr,
    grad_x_ptr,
    n_selected: tl.constexpr,
    axis_size: tl.constexpr,
    outer_size: tl.constexpr,
    inner_size: tl.constexpr,
    OUTER_SHAPE: tl.constexpr,
    INNER_SHAPE: tl.constexpr,
    GRAD_OUTER_STRIDES: tl.constexpr,
    GRAD_INNER_STRIDES: tl.constexpr,
    grad_y_stride_axis,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Process independent segment heads together without changing their sums.

    A lane owns a sorted position; only the first position of each duplicate
    group writes. Short groups accumulate in parallel across lanes. A group
    longer than 16 entries uses the scalar loop below, so a long chain does
    not repeatedly synchronize otherwise idle row lanes. Because BLOCK_ROWS
    is at most 16, a program can contain at most one such long-group head.

    Both paths start at FP32 +0, add in original occurrence order, and cast
    once on store. Integer reductions select work; they never sum gradients.
    """
    tl.static_assert(BLOCK_ROWS <= 16, "a program must own at most 16 sorted positions")
    rows = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    outer, position = rows // n_selected, rows % n_selected
    valid_rows = rows < outer_size * n_selected
    input_row = tl.load(sorted_index_ptr + position, valid_rows, other=-1).to(tl.int64)
    previous_row = tl.load(sorted_index_ptr + position - 1, valid_rows & (position > 0), other=-1)
    segment_head = valid_rows & (input_row != previous_row)
    lookahead = tl.load(
        sorted_index_ptr + position + 16,
        segment_head & (position + 16 < n_selected),
        other=-1,
    )
    long_group = segment_head & (lookahead == input_row)
    short_group = segment_head & ~long_group
    outer_offset = _strided_offset(outer, OUTER_SHAPE, GRAD_OUTER_STRIDES)

    # There is at most one long head, so integer sums only extract its metadata.
    has_long_group = tl.sum(long_group.to(tl.int32), axis=0) != 0
    long_position = tl.sum(tl.where(long_group, position, 0), axis=0)
    long_input_row = tl.sum(tl.where(long_group, input_row, 0), axis=0)
    long_outer = tl.sum(tl.where(long_group, outer, 0), axis=0)
    long_outer_offset = tl.sum(tl.where(long_group, outer_offset, 0), axis=0)

    first_tile = tl.program_id(1)
    n_tiles = tl.cdiv(inner_size, BLOCK_SIZE)
    tile_stride = tl.num_programs(1)
    for tile in range(first_tile, n_tiles, tile_stride):
        cols = tile.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < inner_size
        inner_offsets = _strided_offset(cols, INNER_SHAPE, GRAD_INNER_STRIDES)
        accumulator = tl.full((BLOCK_ROWS, BLOCK_SIZE), 0, tl.float32)
        cursor = position
        active = short_group
        while tl.sum(active.to(tl.int32), axis=0) > 0:
            source_position = tl.load(order_ptr + cursor, active, other=0)
            value = tl.load(
                grad_y_ptr
                + outer_offset[:, None]
                + source_position[:, None] * grad_y_stride_axis
                + inner_offsets[None, :],
                mask=active[:, None] & mask[None, :],
                other=0,
            ).to(tl.float32)
            accumulator = tl.where(active[:, None], accumulator + value, accumulator)
            cursor += 1
            current_row = tl.load(
                sorted_index_ptr + cursor, active & (cursor < n_selected), other=-1
            )
            active = active & (cursor < n_selected) & (current_row == input_row)
        output_offsets = (outer * axis_size + input_row)[:, None] * inner_size + cols[None, :]
        tl.store(
            grad_x_ptr + output_offsets,
            accumulator.to(grad_x_ptr.dtype.element_ty),
            mask=short_group[:, None] & mask[None, :],
        )

        if has_long_group:
            # Load eight consecutive occurrences together, then fold them in
            # order. This shortens the dependent load chain, not the sum chain.
            total = tl.full((BLOCK_SIZE,), 0, tl.float32)
            long_cursor = long_position
            long_current_row = long_input_row
            while (long_cursor < n_selected) & (long_current_row == long_input_row):
                positions = long_cursor + tl.arange(0, 8)
                keys = tl.load(sorted_index_ptr + positions, positions < n_selected, other=-1)
                matches = (positions < n_selected) & (keys == long_input_row)
                sources = tl.load(order_ptr + positions, matches, other=0)
                values = tl.load(
                    grad_y_ptr
                    + long_outer_offset
                    + sources[:, None] * grad_y_stride_axis
                    + inner_offsets[None, :],
                    mask=matches[:, None] & mask[None, :],
                    other=0,
                ).to(tl.float32)
                for offset in tl.static_range(8):
                    prefetched_value = tl.reshape(
                        tl.gather(values, tl.full((1, BLOCK_SIZE), offset, tl.int32), axis=0),
                        (BLOCK_SIZE,),
                    )
                    matched = (
                        tl.sum(
                            tl.where(tl.arange(0, 8) == offset, matches.to(tl.int32), 0),
                            axis=0,
                        )
                        != 0
                    )
                    total = tl.where(matched, total + prefetched_value, total)
                long_cursor += 8
                long_current_row = tl.load(
                    sorted_index_ptr + long_cursor, long_cursor < n_selected, other=-1
                ).to(tl.int64)
            long_output_offsets = (long_outer * axis_size + long_input_row) * inner_size + cols
            tl.store(
                grad_x_ptr + long_output_offsets,
                total.to(grad_x_ptr.dtype.element_ty),
                mask=mask,
            )


@triton.jit
def _index_select_init_buckets_kernel(
    metadata_ptr,
    axis_size: tl.constexpr,
    POSITION_SLOTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Initialize counts, overflow-list heads and the queue length.

    One allocation holds counts [A], positions [P, A], heads [A], a queue
    counter, queued sources [max(1, K // 5)] and overflow links [K]. Position
    slots and links are written before they are read; they need no zero fill.
    """
    source = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(metadata_ptr + source, 0, source < axis_size)
    tl.store(metadata_ptr + (POSITION_SLOTS + 1) * axis_size + source, -1, source < axis_size)
    if tl.program_id(0) == 0:
        tl.store(metadata_ptr + (POSITION_SLOTS + 2) * axis_size, 0)


@triton.jit
def _index_select_build_buckets_kernel(
    index_ptr,
    metadata_ptr,
    axis_size: tl.constexpr,
    n_selected: tl.constexpr,
    index_stride: tl.constexpr,
    POSITION_SLOTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Group integer positions without imposing any floating-point sum order.

    Atomic ranks assign distinct slots, not the order of gradient additions.
    Each group with more than four occurrences enters the queue exactly once.
    Positions beyond the bounded slots form a linked list. Consumers sort the
    collected positions, or scan the original index, before adding gradients.
    """
    counts_ptr = metadata_ptr
    positions_ptr = metadata_ptr + axis_size
    heads_ptr = metadata_ptr + (POSITION_SLOTS + 1) * axis_size
    queue_count_ptr = metadata_ptr + (POSITION_SLOTS + 2) * axis_size
    queue_ptr = queue_count_ptr + 1
    links_ptr = queue_ptr + max(1, n_selected // 5)
    position = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = position < n_selected
    source = tl.load(index_ptr + position.to(tl.int64) * index_stride, valid, other=0).to(tl.int64)
    rank = tl.atomic_add(counts_ptr + source, 1, valid, sem="relaxed")
    tl.store(
        positions_ptr + rank.to(tl.int64) * axis_size + source,
        position,
        valid & (rank < POSITION_SLOTS),
    )
    overflow = valid & (rank >= POSITION_SLOTS)
    previous = tl.atomic_xchg(heads_ptr + source, position, overflow, sem="relaxed")
    tl.store(links_ptr + position, previous, overflow)
    enqueue = valid & (rank == 4)
    queue_position = tl.atomic_add(
        queue_count_ptr + tl.full((BLOCK_SIZE,), 0, tl.int32), 1, enqueue, sem="relaxed"
    )
    tl.store(queue_ptr + queue_position, source.to(tl.int32), enqueue)


@triton.jit
def _index_select_bwd_short_groups(
    counts_ptr,
    positions_ptr,
    grad_y_ptr,
    grad_x_ptr,
    row_tile,
    column_tile,
    axis_size: tl.constexpr,
    outer_size: tl.constexpr,
    inner_size: tl.constexpr,
    OUTER_SHAPE: tl.constexpr,
    INNER_SHAPE: tl.constexpr,
    GRAD_OUTER_STRIDES: tl.constexpr,
    GRAD_INNER_STRIDES: tl.constexpr,
    grad_y_stride_axis: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Own zero-to-four-occurrence source rows; long groups are never written.

    Integer compare/exchanges recover original occurrence order for four slots.
    A tile containing only zero/one-occurrence rows avoids that work. Even a
    single occurrence starts with FP32 +0, preserving the signed-zero contract.
    Disjoint stores allow short and long work to share one kernel launch.
    """
    rows = tl.cast(row_tile, tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    outer, source = rows // axis_size, rows % axis_size
    valid_rows = rows < outer_size * axis_size
    counts = tl.load(counts_ptr + source, valid_rows, other=0)
    cols = tl.cast(column_tile, tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    outer_offsets = _strided_offset(outer, OUTER_SHAPE, GRAD_OUTER_STRIDES)
    inner_offsets = _strided_offset(cols, INNER_SHAPE, GRAD_INNER_STRIDES)
    total = tl.full((BLOCK_ROWS, BLOCK_SIZE), 0, tl.float32)
    if tl.max(counts, axis=0) <= 1:
        position = tl.load(positions_ptr + source, valid_rows & (counts > 0), other=0)
        value = tl.load(
            grad_y_ptr
            + outer_offsets[:, None]
            + position[:, None].to(tl.int64) * grad_y_stride_axis
            + inner_offsets[None, :],
            valid_rows[:, None] & (counts[:, None] > 0) & (cols[None, :] < inner_size),
            other=0,
        ).to(tl.float32)
        total = total + value
    else:
        short = valid_rows & (counts <= 4)
        p0 = tl.load(positions_ptr + source, short & (counts > 0), other=2147483647)
        p1 = tl.load(positions_ptr + axis_size + source, short & (counts > 1), other=2147483647)
        p2 = tl.load(
            positions_ptr + 2 * axis_size + source,
            short & (counts > 2),
            other=2147483647,
        )
        p3 = tl.load(
            positions_ptr + 3 * axis_size + source,
            short & (counts > 3),
            other=2147483647,
        )
        p0, p1 = tl.minimum(p0, p1), tl.maximum(p0, p1)
        p2, p3 = tl.minimum(p2, p3), tl.maximum(p2, p3)
        p0, p2 = tl.minimum(p0, p2), tl.maximum(p0, p2)
        p1, p3 = tl.minimum(p1, p3), tl.maximum(p1, p3)
        p1, p2 = tl.minimum(p1, p2), tl.maximum(p1, p2)
        positions = (p0, p1, p2, p3)
        for slot in tl.static_range(4):
            active = short & (slot < counts)
            value = tl.load(
                grad_y_ptr
                + outer_offsets[:, None]
                + positions[slot][:, None].to(tl.int64) * grad_y_stride_axis
                + inner_offsets[None, :],
                active[:, None] & (cols[None, :] < inner_size),
                other=0,
            ).to(tl.float32)
            total = tl.where(active[:, None], total + value, total)
    tl.store(
        grad_x_ptr + rows[:, None] * inner_size + cols[None, :],
        total.to(grad_x_ptr.dtype.element_ty),
        valid_rows[:, None] & (counts[:, None] <= 4) & (cols[None, :] < inner_size),
    )


@triton.jit
def _index_select_fold_positions(
    grad_y_ptr,
    positions,
    count,
    grad_offsets,
    valid_columns,
    total,
    grad_y_stride_axis: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Prefetch eight occurrences, then carry the same FP32 sum through them.

    Integer positions are already sorted. Each load batch changes memory
    scheduling only: there is no partial floating sum and no reassociation.
    """
    POSITION_BLOCK: tl.constexpr = positions.shape[0]
    for window in range(tl.cdiv(count, 8)):
        ranks = window * 8 + tl.arange(0, 8)
        selected = tl.gather(positions, tl.minimum(ranks, POSITION_BLOCK - 1), axis=0)
        values = tl.load(
            grad_y_ptr
            + selected[:, None].to(tl.int64) * grad_y_stride_axis
            + grad_offsets[None, :],
            (ranks[:, None] < count) & valid_columns[None, :],
            other=0,
        ).to(tl.float32)
        for offset in tl.static_range(8):
            value = tl.reshape(
                tl.gather(values, tl.full((1, BLOCK_SIZE), offset, tl.int32), axis=0),
                (BLOCK_SIZE,),
            )
            total = tl.where(window * 8 + offset < count, total + value, total)
    return total


@triton.jit
def _index_select_bucket_positions(
    positions_ptr,
    heads_ptr,
    links_ptr,
    source,
    count,
    axis_size: tl.constexpr,
    POSITION_SLOTS: tl.constexpr,
    POSITION_BLOCK: tl.constexpr,
):
    """Recover a bounded group's positions and sort only its integer metadata."""
    ranks = tl.arange(0, POSITION_BLOCK)
    positions = tl.load(
        positions_ptr + ranks.to(tl.int64) * axis_size + source,
        (ranks < count) & (ranks < POSITION_SLOTS),
        other=2147483647,
    )
    if POSITION_SLOTS == 4:
        cursor = tl.load(heads_ptr + source)
        for slot in range(4, count):
            positions = tl.where(ranks == slot, cursor, positions)
            cursor = tl.load(links_ptr + cursor)
    positions = tl.sort(positions, descending=False)
    return positions


@triton.jit
def _index_select_bwd_long_groups(
    metadata_ptr,
    index_ptr,
    grad_y_ptr,
    grad_x_ptr,
    first_work,
    column_tile,
    axis_size: tl.constexpr,
    n_selected: tl.constexpr,
    outer_size: tl.constexpr,
    inner_size: tl.constexpr,
    OUTER_SHAPE: tl.constexpr,
    INNER_SHAPE: tl.constexpr,
    GRAD_OUTER_STRIDES: tl.constexpr,
    GRAD_INNER_STRIDES: tl.constexpr,
    grad_y_stride_axis: tl.constexpr,
    index_stride: tl.constexpr,
    POSITION_SLOTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WORK_STRIDE: tl.constexpr,
):
    """Consume each queued source in original index order, including overflow.

    Most groups load their integer positions in parallel and sort locally.
    With four-slot metadata, collect the remaining positions from the list.
    Groups larger than 128 scan bounded original-index tiles. The FP32 total
    is carried across all tiles, so that fallback has exactly the same sum.
    """
    positions_ptr = metadata_ptr + axis_size
    heads_ptr = metadata_ptr + (POSITION_SLOTS + 1) * axis_size
    queue_count_ptr = metadata_ptr + (POSITION_SLOTS + 2) * axis_size
    queue_ptr = queue_count_ptr + 1
    links_ptr = queue_ptr + max(1, n_selected // 5)
    queue_length = tl.load(queue_count_ptr)
    columns = tl.cast(column_tile, tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    outer, cols = columns // inner_size, columns % inner_size
    valid_columns = columns < outer_size * inner_size
    grad_offsets = _strided_offset(outer, OUTER_SHAPE, GRAD_OUTER_STRIDES)
    grad_offsets += _strided_offset(cols, INNER_SHAPE, GRAD_INNER_STRIDES)
    for work in range(first_work, queue_length, WORK_STRIDE):
        source = tl.load(queue_ptr + work).to(tl.int64)
        count = tl.load(metadata_ptr + source)
        total = tl.full((BLOCK_SIZE,), 0, tl.float32)
        if count <= 16:
            positions = _index_select_bucket_positions(
                positions_ptr,
                heads_ptr,
                links_ptr,
                source,
                count,
                axis_size,
                POSITION_SLOTS,
                16,
            )
            total = _index_select_fold_positions(
                grad_y_ptr,
                positions,
                count,
                grad_offsets,
                valid_columns,
                total,
                grad_y_stride_axis,
                BLOCK_SIZE,
            )
        elif count <= 128:
            positions = _index_select_bucket_positions(
                positions_ptr,
                heads_ptr,
                links_ptr,
                source,
                count,
                axis_size,
                POSITION_SLOTS,
                128,
            )
            total = _index_select_fold_positions(
                grad_y_ptr,
                positions,
                count,
                grad_offsets,
                valid_columns,
                total,
                grad_y_stride_axis,
                BLOCK_SIZE,
            )
        else:
            for tile in range(tl.cdiv(n_selected, 256)):
                ranks = tile * 256 + tl.arange(0, 256)
                keys = tl.load(
                    index_ptr + ranks.to(tl.int64) * index_stride,
                    ranks < n_selected,
                    other=-1,
                )
                matches = (ranks < n_selected) & (keys == source)
                positions = tl.sort(tl.where(matches, ranks, 2147483647), descending=False)
                count_in_tile = tl.sum(matches.to(tl.int32), axis=0)
                total = _index_select_fold_positions(
                    grad_y_ptr,
                    positions,
                    count_in_tile,
                    grad_offsets,
                    valid_columns,
                    total,
                    grad_y_stride_axis,
                    BLOCK_SIZE,
                )
        tl.store(
            grad_x_ptr + (outer * axis_size + source) * inner_size + cols,
            total.to(grad_x_ptr.dtype.element_ty),
            valid_columns,
        )


@triton.jit
def _index_select_bwd_buckets_kernel(
    metadata_ptr,
    index_ptr,
    grad_y_ptr,
    grad_x_ptr,
    axis_size: tl.constexpr,
    n_selected: tl.constexpr,
    outer_size: tl.constexpr,
    inner_size: tl.constexpr,
    OUTER_SHAPE: tl.constexpr,
    INNER_SHAPE: tl.constexpr,
    GRAD_OUTER_STRIDES: tl.constexpr,
    GRAD_INNER_STRIDES: tl.constexpr,
    grad_y_stride_axis: tl.constexpr,
    index_stride: tl.constexpr,
    POSITION_SLOTS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LONG_BLOCK_SIZE: tl.constexpr,
    QUEUE_PROGRAMS: tl.constexpr,
    PROCESS_SHORT: tl.constexpr,
    PROCESS_LONG: tl.constexpr,
):
    """Schedule disjoint source coordinates; optionally separate heavy kernels.

    Small/mid-size outputs combine short and long work into one launch. Large
    outputs use two specializations so the heavier sorting path does not set
    the register/shared-memory requirements of every short-copy program.
    """
    program = tl.program_id(0).to(tl.int64)
    SHORT_COLUMNS: tl.constexpr = tl.cdiv(inner_size, BLOCK_SIZE)
    SHORT_PROGRAMS: tl.constexpr = tl.cdiv(outer_size * axis_size, BLOCK_ROWS) * SHORT_COLUMNS
    LONG_COLUMNS: tl.constexpr = tl.cdiv(outer_size * inner_size, LONG_BLOCK_SIZE)
    if PROCESS_SHORT:
        if program < SHORT_PROGRAMS:
            _index_select_bwd_short_groups(
                metadata_ptr,
                metadata_ptr + axis_size,
                grad_y_ptr,
                grad_x_ptr,
                program // SHORT_COLUMNS,
                program % SHORT_COLUMNS,
                axis_size,
                outer_size,
                inner_size,
                OUTER_SHAPE,
                INNER_SHAPE,
                GRAD_OUTER_STRIDES,
                GRAD_INNER_STRIDES,
                grad_y_stride_axis,
                BLOCK_ROWS,
                BLOCK_SIZE,
            )
    if PROCESS_LONG:
        work = program - SHORT_PROGRAMS if PROCESS_SHORT else program
        if work >= 0:
            _index_select_bwd_long_groups(
                metadata_ptr,
                index_ptr,
                grad_y_ptr,
                grad_x_ptr,
                work // LONG_COLUMNS,
                work % LONG_COLUMNS,
                axis_size,
                n_selected,
                outer_size,
                inner_size,
                OUTER_SHAPE,
                INNER_SHAPE,
                GRAD_OUTER_STRIDES,
                GRAD_INNER_STRIDES,
                grad_y_stride_axis,
                index_stride,
                POSITION_SLOTS,
                LONG_BLOCK_SIZE,
                QUEUE_PROGRAMS,
            )


def _validate_inputs(x: Tensor, index: Tensor, dim: int) -> int:
    """Check the device and metadata; return a nonnegative axis.

    Accept strided tensors and empty dimensions. The forward launch checks
    index values on device without copying them to CPU.
    This backend owns its checks independently of the PyTorch reference.
    """
    if x.device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError("index_select requires NVIDIA CUDA; no fallback")
    major, minor = torch.cuda.get_device_capability(x.device)
    if 10 * major + minor < _MIN_CUDA_SM:
        raise RuntimeError(f"index_select requires CUDA SM{_MIN_CUDA_SM}+")
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
    return dim


def _launch_index_select_fwd(x: Tensor, index: Tensor, dim: int) -> Tensor:
    """Allocate the selected shape and copy slices on the input device's stream."""
    outer_shape, inner_shape = x.shape[:dim], x.shape[dim + 1 :]
    outer_size, inner_size = prod(outer_shape), prod(inner_shape)
    n_selected = index.numel()
    output_shape = (*outer_shape, n_selected, *inner_shape)
    y = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    device_module = torch.get_device_module(x.device)
    with device_module.device(x.device):
        if y.numel() == 0:
            if n_selected:
                _index_select_check_bounds_kernel[(triton.cdiv(n_selected, _BLOCK_SIZE),)](
                    index,
                    n_selected,
                    x.shape[dim],
                    index.stride(0),
                    BLOCK_SIZE=_BLOCK_SIZE,
                    debug=True,
                )
            return y
        if x.is_contiguous():
            # Wider output tiles amortize address work on large copies. This
            # changes only ownership of copied elements, never arithmetic.
            block_size = 256
            if inner_size >= 16 and y.numel() >= 2**18:
                block_size = 1024
            if y.numel() >= 2**24 or (inner_size == 1 and outer_size > 1):
                block_size = 2048
            grid = (
                outer_size,
                min(triton.cdiv(n_selected * inner_size, block_size), _MAX_GRID_Y),
            )
            _index_select_fwd_contiguous_kernel[grid](
                x,
                index,
                y,
                x.shape[dim],
                n_selected,
                inner_size,
                index.stride(0),
                block_size,
                num_warps=4,
                debug=True,
            )
            return y
        block_size = min(triton.next_power_of_2(inner_size), _BLOCK_SIZE)
        # Pack independent narrow slices; this does not define a reduction tree.
        block_rows = max(1, 512 // block_size) if inner_size > 1 else 128
        grid = (
            triton.cdiv(outer_size * n_selected, block_rows),
            min(triton.cdiv(inner_size, block_size), _MAX_GRID_Y),
        )
        _index_select_fwd_kernel[grid](
            x,
            index,
            y,
            n_selected=n_selected,
            axis_size=x.shape[dim],
            outer_size=outer_size,
            inner_size=inner_size,
            OUTER_SHAPE=tuple(outer_shape),
            INNER_SHAPE=tuple(inner_shape),
            X_OUTER_STRIDES=x.stride()[:dim],
            X_INNER_STRIDES=x.stride()[dim + 1 :],
            x_stride_axis=x.stride(dim),
            index_stride=index.stride(0),
            BLOCK_SIZE=block_size,
            BLOCK_ROWS=block_rows,
            num_warps=4,
            debug=True,
        )
    return y


def _sort_index_occurrences(index: Tensor, axis_size: int) -> tuple[Tensor, Tensor]:
    """Return sorted input indices and their stable original positions.

    Small vectors use one integer-sort launch. Only use packed int32 keys when
    the shape proves they cannot overflow; otherwise keep PyTorch's stable sort.
    The larger-vector path consumes both returned tensors directly, avoiding
    argsort followed by a separate index_select copy. No GPU values reach CPU.
    """
    n_selected = index.numel()
    # Triton 3.4's sort does not compile a one-element vector; pad it to two.
    block_size = max(2, triton.next_power_of_2(n_selected))
    if n_selected > _SMALL_SORT_LIMIT or axis_size > (2**31 - 1) // block_size:
        return torch.sort(index, stable=True)

    sorted_index = torch.empty((n_selected,), device=index.device, dtype=index.dtype)
    order = torch.empty((n_selected,), device=index.device, dtype=torch.int64)
    num_warps = 8 if block_size >= 512 else 4 if block_size >= 128 else 1
    _index_select_sort_kernel[(1,)](
        index,
        sorted_index,
        order,
        n_selected=n_selected,
        index_stride=index.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return sorted_index, order


def _dense_bucket_slots(axis_size: int, n_selected: int) -> int:
    """Choose bounded metadata from shapes only; zero means stable-sort fallback.

    Dense indexing benefits from direct position slots. Cap the entire integer
    workspace at 128 MiB and use four slots when 128 would exceed that cap.
    No input values, uniqueness promises or cached index contents are involved.
    """
    if axis_size > 8 * n_selected or axis_size >= 2**28 or n_selected >= 2**30:
        return 0
    for slots in (128, 4):
        elements = (slots + 2) * axis_size + 1 + max(1, n_selected // 5) + n_selected
        if elements * 4 <= _MAX_BUCKET_BYTES:
            return slots
    return 0


def _launch_index_select_bwd_buckets(
    grad_y: Tensor,
    index: Tensor,
    x_shape: tuple[int, ...],
    dim: int,
    position_slots: int,
) -> Tensor:
    """Build integer metadata, then write gradients with disjoint ownership.

    Usually three launches and two allocations suffice. Large metadata domains
    split the final launch into short/long kernels to keep short work's register
    usage low. All paths use the same original-order FP32 accumulation. Called
    inside the input device guard; launches share the caller's current stream.
    """
    outer_shape, inner_shape = x_shape[:dim], x_shape[dim + 1 :]
    outer_size, inner_size = prod(outer_shape), prod(inner_shape)
    axis_size, n_selected = x_shape[dim], index.numel()
    queue_capacity = max(1, n_selected // 5)
    metadata_size = (position_slots + 2) * axis_size + 1 + queue_capacity + n_selected
    metadata = torch.empty((metadata_size,), device=index.device, dtype=torch.int32)
    grad_x = torch.empty(x_shape, device=grad_y.device, dtype=grad_y.dtype)
    _index_select_init_buckets_kernel[(triton.cdiv(axis_size, 512),)](
        metadata,
        axis_size,
        position_slots,
        512,
    )
    _index_select_build_buckets_kernel[(triton.cdiv(n_selected, 256),)](
        index,
        metadata,
        axis_size,
        n_selected,
        index.stride(0),
        position_slots,
        256,
    )
    block_size = min(triton.next_power_of_2(inner_size), 256)
    if inner_size == 1:
        block_rows = 128
    elif position_slots == 4 or block_size == 256:
        block_rows = max(8, 512 // block_size)
    else:
        block_rows = max(16, 512 // block_size)
    long_block_size = min(triton.next_power_of_2(outer_size * inner_size), 128)
    queue_programs = min(queue_capacity, 1024)
    short_programs = triton.cdiv(outer_size * axis_size, block_rows) * triton.cdiv(
        inner_size, block_size
    )
    long_programs = queue_programs * triton.cdiv(outer_size * inner_size, long_block_size)
    arguments = (
        metadata,
        index,
        grad_y,
        grad_x,
        axis_size,
        n_selected,
        outer_size,
        inner_size,
        tuple(outer_shape),
        tuple(inner_shape),
        grad_y.stride()[:dim],
        grad_y.stride()[dim + 1 :],
        grad_y.stride(dim),
        index.stride(0),
        position_slots,
        block_rows,
        block_size,
        long_block_size,
        queue_programs,
    )
    if position_slots == 128:
        _index_select_bwd_buckets_kernel[(short_programs + long_programs,)](
            *arguments,
            True,
            True,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        _index_select_bwd_buckets_kernel[(short_programs,)](
            *arguments,
            True,
            False,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _index_select_bwd_buckets_kernel[(long_programs,)](
            *arguments,
            False,
            True,
            num_warps=4,
            enable_fp_fusion=False,
        )
    record_backward(
        "index_select",
        kernel_id="+".join(
            __name__ + "." + name
            for name in (
                "_index_select_init_buckets_kernel",
                "_index_select_build_buckets_kernel",
                "_index_select_bwd_buckets_kernel",
            )
        ),
        impl=BACKWARD_IMPL,
        family="triton",
    )
    return grad_x


def _launch_index_select_bwd(
    grad_y: Tensor, index: Tensor, x_shape: tuple[int, ...], dim: int
) -> Tensor:
    """Choose index grouping from shape metadata, preserving one FP32 sum order.

    Small selections gather directly. Dense axes use integer buckets; sparse or
    very large index domains retain stable sorting to bound temporary storage.
    No device values are copied to CPU for dispatch, and no index cache is kept.
    """
    if grad_y.numel() == 0:
        grad_x = torch.zeros(x_shape, device=grad_y.device, dtype=grad_y.dtype)
        record_backward(
            "index_select",
            kernel_id="torch.zeros",
            impl="empty_zero_gradient",
            family="triton",
        )
        return grad_x

    outer_shape, inner_shape = x_shape[:dim], x_shape[dim + 1 :]
    outer_size, inner_size = prod(outer_shape), prod(inner_shape)
    device_module = torch.get_device_module(grad_y.device)
    with device_module.device(grad_y.device):
        n_selected = index.numel()
        block_size = min(triton.next_power_of_2(inner_size), _BLOCK_SIZE)
        tiny_selection = n_selected <= 64 and x_shape[dim] <= 256 and outer_size <= 16
        bounded_scan = (
            n_selected <= 1024
            and 1 < inner_size <= 128
            and outer_size <= 16
            and outer_size * x_shape[dim] * n_selected <= 2**24
        )
        if tiny_selection or bounded_scan:
            grad_x = torch.empty(x_shape, device=grad_y.device, dtype=grad_y.dtype)
            grid = (
                outer_size * x_shape[dim],
                min(triton.cdiv(inner_size, block_size), _MAX_GRID_Y),
            )
            _index_select_bwd_small_kernel[grid](
                grad_y,
                index,
                grad_x,
                n_selected=n_selected,
                axis_size=x_shape[dim],
                inner_size=inner_size,
                OUTER_SHAPE=tuple(outer_shape),
                INNER_SHAPE=tuple(inner_shape),
                GRAD_OUTER_STRIDES=grad_y.stride()[:dim],
                GRAD_INNER_STRIDES=grad_y.stride()[dim + 1 :],
                grad_y_stride_axis=grad_y.stride(dim),
                index_stride=index.stride(0),
                INDEX_BLOCK=triton.next_power_of_2(n_selected),
                BLOCK_SIZE=block_size,
                num_warps=4,
                enable_fp_fusion=False,
            )
            record_backward(
                "index_select",
                kernel_id=__name__ + "._index_select_bwd_small_kernel",
                impl=BACKWARD_IMPL,
                family="triton",
            )
            return grad_x
        position_slots = _dense_bucket_slots(x_shape[dim], n_selected)
        if position_slots:
            return _launch_index_select_bwd_buckets(grad_y, index, x_shape, dim, position_slots)
        grad_x = torch.zeros(x_shape, device=grad_y.device, dtype=grad_y.dtype)
        sorted_index, order = _sort_index_occurrences(index, x_shape[dim])
        # Narrow slices share a program; wide slices already fill its lanes.
        block_rows = 16 if inner_size == 1 else 4 if inner_size <= 128 else 1
        if inner_size > 128:
            num_warps = 2
        elif inner_size >= 64:
            # Four row lanes plus eight prefetched occurrences need more lanes.
            num_warps = 4
        else:
            num_warps = 1
        grid = (
            triton.cdiv(outer_size * n_selected, block_rows),
            min(triton.cdiv(inner_size, block_size), _MAX_GRID_Y),
        )
        _index_select_bwd_kernel[grid](
            sorted_index,
            order,
            grad_y,
            grad_x,
            n_selected=n_selected,
            axis_size=x_shape[dim],
            outer_size=outer_size,
            inner_size=inner_size,
            OUTER_SHAPE=tuple(outer_shape),
            INNER_SHAPE=tuple(inner_shape),
            GRAD_OUTER_STRIDES=grad_y.stride()[:dim],
            GRAD_INNER_STRIDES=grad_y.stride()[dim + 1 :],
            grad_y_stride_axis=grad_y.stride(dim),
            BLOCK_SIZE=block_size,
            BLOCK_ROWS=block_rows,
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
    record_backward(
        "index_select",
        kernel_id=BACKWARD_KERNEL_ID,
        impl=BACKWARD_IMPL,
        family="triton",
    )
    return grad_x


class _IndexSelectTritonFunction(torch.autograd.Function):
    """Save indices, input shape and selected axis for the first-order gradient."""

    @staticmethod
    def forward(ctx, x: Tensor, index: Tensor, dim: int) -> Tensor:
        y = _launch_index_select_fwd(x, index, dim)
        ctx.save_for_backward(index)
        ctx.x_shape = x.shape
        ctx.dim = dim
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y: Tensor) -> tuple[Tensor, None, None]:
        (index,) = ctx.saved_tensors
        grad_x = _launch_index_select_bwd(grad_y, index, ctx.x_shape, ctx.dim)
        return grad_x, None, None


class TritonIndexSelectOp:
    """Select slices along an explicit axis with an ordered input gradient.

    Args:
        x: Rank >= 1 FP32/FP16/BF16 tensor on NVIDIA CUDA SM80+.
        index: Same-device int32/int64 vector ``[K]`` of indices along dim.
            Repeated indices are allowed; their occurrence order is preserved.
        dim: Required keyword axis. Negative axes count from the end of x.

    Output replaces x.shape[dim] by K and preserves input dtype and copied bits.
    Indices are shared across all other coordinates; negative indices are invalid.
    Strided inputs, indices and upstream gradients are supported. Forward uses
    a 2-D grid and writes contiguous output. Backward assigns one owner to each
    source coordinate, groups its occurrences, and sums upstreams in increasing output
    position using FP32 additions from +0, casting once on store. Each coordinate
    outside dim is independent. No floating-point atomics are used.

    Repeatability holds for a fixed occurrence order; batching independent slices
    does not alter it. Reordering duplicates or splitting their sum across backward
    calls can change rounding. First-order autograd only.
    ROCm is not qualified and fails closed.
    """

    op_class = "reduction"
    is_batch_invariant = True
    kernel_id = KERNEL_ID
    backward_impl = BACKWARD_IMPL

    def __call__(self, x: Tensor, index: Tensor, *, dim: int) -> Tensor:
        return self.forward(x, index, dim=dim)

    def forward(self, x: Tensor, index: Tensor, *, dim: int) -> Tensor:
        """Copy x's indexed slices along dim; record a first-order input VJP."""
        dim = _validate_inputs(x, index, dim)
        return _IndexSelectTritonFunction.apply(x, index, dim)
