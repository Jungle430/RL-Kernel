# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only launchers; compare the public ordered helpers with native tl.sum."""

import torch
import triton
import triton.language as tl

from rl_engine.backends.shared.triton import reduction as ordered


@triton.jit
def _row_sum(
    x,
    out,
    stride,
    LENGTH: tl.constexpr,
    BLOCK: tl.constexpr,
    ORDERED: tl.constexpr,
    GROUP: tl.constexpr,
    SECTIONS: tl.constexpr,
    PIPELINE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if ORDERED:
        total = ordered.sum(
            x + row * stride,
            LENGTH,
            COMPUTE_DTYPE=tl.float32,
            BLOCK_SIZE=BLOCK,
            MAX_GROUP_SIZE=GROUP,
            SECTION_COUNT=SECTIONS,
            PIPELINE_STAGES=PIPELINE,
        )
    else:
        cols = tl.arange(0, BLOCK)
        values = tl.load(x + row * stride + cols, cols < LENGTH, other=0).to(tl.float32)
        total = tl.sum(values, axis=0)
    tl.store(out + row, total)


@triton.jit
def _partial_sum(
    x,
    partials,
    stride,
    LENGTH: tl.constexpr,
    BLOCK: tl.constexpr,
    ORDERED: tl.constexpr,
    GROUP: tl.constexpr,
    SECTIONS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x + row * stride + cols, cols < LENGTH, other=0).to(tl.float32)
    if ORDERED:
        total = ordered.sum(values, MAX_GROUP_SIZE=GROUP, SECTION_COUNT=SECTIONS)
    else:
        total = tl.sum(values, axis=0)
    tl.store(partials + row * tl.cdiv(LENGTH, BLOCK) + tile, total)


@triton.jit
def _merge_sum(
    partials,
    out,
    COUNT: tl.constexpr,
    WIDTH: tl.constexpr,
    ORDERED: tl.constexpr,
    UNROLL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if ORDERED:
        total = ordered.sum_partials(partials + row * COUNT, COUNT, MERGE_UNROLL=UNROLL)
    else:
        cols = tl.arange(0, WIDTH)
        values = tl.load(partials + row * COUNT + cols, cols < COUNT, other=0)
        total = tl.sum(values, axis=0)
    tl.store(out + row, total)


def make_call(x, out, length, provider, tree):
    """Return a preallocated call and explicit scratch bytes; closures retain scratch.

    All providers reduce the same valid prefix to FP32. The plain Triton row
    baseline uses a full-row tl.sum; its parallel baseline uses tl.sum twice.
    Those baselines have no fixed-tree equivalence requirement.
    """
    selected = x[:, :length]
    family = provider["family"]
    if family == "torch":
        return lambda: torch.sum(selected, dim=-1, dtype=torch.float32, out=out), 0
    options = dict(num_warps=provider["warps"], enable_fp_fusion=False)
    if torch.version.hip is not None:
        options["allow_flush_denorm"] = False
    rows, stride = x.shape[0], x.stride(0)
    is_ordered = family.startswith("ordered_")
    block = triton.next_power_of_2(length) if family == "tl_row" else tree["block_size"]
    if family in ("tl_row", "ordered_serial"):

        def call():
            _row_sum[(rows,)](
                x,
                out,
                stride,
                length,
                block,
                is_ordered,
                tree["max_group_size"],
                tree["section_count"],
                provider["pipeline"],
                **options,
            )

        return call, 0
    count = triton.cdiv(length, block)
    partials = torch.empty((rows, count), device=x.device, dtype=torch.float32)

    def call():
        _partial_sum[(rows, count)](
            x,
            partials,
            stride,
            length,
            block,
            is_ordered,
            tree["max_group_size"],
            tree["section_count"],
            **options,
        )
        _merge_sum[(rows,)](
            partials,
            out,
            count,
            triton.next_power_of_2(count),
            is_ordered,
            provider["merge_unroll"],
            **options,
        )

    return call, partials.numel() * partials.element_size()
