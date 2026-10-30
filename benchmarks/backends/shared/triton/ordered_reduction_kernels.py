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
    PIPELINE: tl.constexpr,
    TILE_UNROLL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if ORDERED:
        total = ordered.sum(
            x + row * stride,
            LENGTH,
            COMPUTE_DTYPE=tl.float32,
            BLOCK_SIZE=BLOCK,
            MAX_GROUP_SIZE=GROUP,
            PIPELINE_STAGES=PIPELINE,
            TILE_UNROLL=TILE_UNROLL,
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
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x + row * stride + cols, cols < LENGTH, other=0).to(tl.float32)
    if ORDERED:
        total = ordered.sum(values, MAX_GROUP_SIZE=GROUP)
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
    if family in ("ordered_gluon", "ordered_gluon_parallel"):
        from rl_engine.platforms.hardware import is_h100

        if not is_h100(
            device_type=x.device.type,
            device_name=torch.cuda.get_device_name(x.device),
            capability=torch.cuda.get_device_capability(x.device),
            is_rocm=torch.version.hip is not None,
        ):
            raise RuntimeError("ordered_gluon is only qualified for NVIDIA H100")
        # Select the enclosing kernel on the host. An ordinary Triton JIT caller
        # cannot change language merely by importing a Gluon reduction helper.
        from benchmarks.backends.cuda.ordered_reduction_kernels import (
            merge_sum,
            partial_sum,
            row_sum,
        )

        if family == "ordered_gluon_parallel":
            count = triton.cdiv(length, tree["block_size"])
            partials = torch.empty((x.shape[0], count), device=x.device, dtype=torch.float32)

            def call():
                partial_sum[(x.shape[0], count)](
                    x,
                    partials,
                    x.stride(0),
                    length,
                    tree["block_size"],
                    tree["max_group_size"],
                    provider["elements_per_thread"],
                    num_warps=provider["warps"],
                    enable_fp_fusion=False,
                )
                merge_sum[(x.shape[0],)](
                    partials,
                    out,
                    count,
                    num_warps=provider["merge_warps"],
                    enable_fp_fusion=False,
                )

            return call, partials.numel() * partials.element_size()

        def call():
            row_sum[(x.shape[0],)](
                x,
                out,
                x.stride(0),
                length,
                tree["block_size"],
                tree["max_group_size"],
                provider["tiles_per_batch"],
                provider["elements_per_thread"],
                num_warps=provider["warps"],
                enable_fp_fusion=False,
            )

        return call, 0
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
                provider["pipeline"],
                provider["tile_unroll"],
                **options,
            )

        return call, 0
    count = triton.cdiv(length, block)
    partials = torch.empty((rows, count), device=x.device, dtype=torch.float32)
    merge_options = dict(options, num_warps=provider.get("merge_warps", provider["warps"]))

    def call():
        _partial_sum[(rows, count)](
            x,
            partials,
            stride,
            length,
            block,
            is_ordered,
            tree["max_group_size"],
            **options,
        )
        _merge_sum[(rows,)](
            partials,
            out,
            count,
            triton.next_power_of_2(count),
            is_ordered,
            provider["merge_unroll"],
            **merge_options,
        )

    return call, partials.numel() * partials.element_size()
