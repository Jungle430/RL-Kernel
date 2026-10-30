# SPDX-License-Identifier: Apache-2.0
"""Fixed-order reduction conformance, independent of model/operator tests."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from rl_engine.reference.reduction.ordered_reduce import ordered_tile_reduce
from tests.helpers.ordered_reduce_reference import ordered_reduce
from tests.helpers.tensor_checks import assert_bitwise

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_TILE_WIDTHS = tuple(1 << exponent for exponent in range(14))
_TREE_CONFIG = {"max_group_size": 8}
"""Eight-value sequential groups, then one tree over all group totals."""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize(
    "dtype",
    (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    ),
)
@pytest.mark.parametrize("num_warps", (1, 2, 4))
@pytest.mark.parametrize(
    "group_size, row_padding, storage_offset",
    ((1, 0, 0), (2, 1, 1), (8, 0, 0), (8, 1, 0), (32, 8, 1)),
)
def test_group_selection_copies_original_bits(
    dtype, num_warps, group_size, row_padding, storage_offset
):
    """Copy exact bits under aligned, odd-stride and offset-pointer layouts.

    Every column must survive extraction. Raw floating payloads include negative
    zero, infinities, distinct NaNs, a low mantissa bit and the least subnormal.
    Integers cover all supported widths, including values beyond FP64's exact
    integer range. The result must keep both the input dtype and its raw bits.
    """
    from tests.helpers.ordered_reduce_kernels import group_column_probe

    floating_payloads = {
        torch.float16: (
            torch.uint16,
            [0x8000, 0x7C00, 0xFC00, 0x7E11, 0xFE22, 0, 0x3C01, 1],
        ),
        torch.bfloat16: (
            torch.uint16,
            [0x8000, 0x7F80, 0xFF80, 0x7FC1, 0xFFC2, 0, 0x3F81, 1],
        ),
        torch.float32: (
            torch.uint32,
            [
                0x80000000,
                0x7F800000,
                0xFF800000,
                0x7FC01234,
                0xFFC05678,
                0,
                0x3F800001,
                1,
            ],
        ),
        torch.float64: (
            torch.uint64,
            [
                0x8000000000000000,
                0x7FF0000000000000,
                0xFFF0000000000000,
                0x7FF8000000001234,
                0xFFF8000000005678,
                0,
                0x3FF0000000000001,
                1,
            ],
        ),
    }
    if dtype.is_floating_point:
        bit_dtype, payload = floating_payloads[dtype]
        pattern = torch.tensor(payload, dtype=bit_dtype).view(dtype)
    elif dtype == torch.bool:
        pattern = torch.tensor([True, False, True, False, False, True, False, True])
    elif dtype == torch.int64:
        pattern = torch.tensor([2**60 + 1, -(2**60) + 3, -1, 0, 1, 7, 2**53 + 1, 31])
    else:
        info = torch.iinfo(dtype)
        pattern = torch.tensor([info.min, info.max, 0, 1, 2, 3, 17, 29], dtype=dtype)
    source = pattern.repeat((group_size + 7) // 8)[:group_size].repeat(8, 1)
    source[1::2] = source[1::2].flip(-1)
    row_stride = group_size + row_padding
    storage = torch.empty(8 * row_stride + storage_offset, dtype=dtype, device="cuda")
    values = torch.as_strided(storage, (8, group_size), (row_stride, 1), storage_offset)
    values.copy_(source)
    actual = torch.empty((8, group_size), dtype=dtype, device="cuda")
    group_column_probe[(1,)](
        values,
        actual,
        values.stride(0),
        GROUP_SIZE=group_size,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    assert_bitwise(actual, values, "group column extraction preserves the source bits")


def test_ordered_tree_accepts_interpreter_constexpr_sizes():
    """Exercise the single halving loop with CPU tensors in a fresh interpreter process.

    TRITON_INTERPRET must be set before importing the JIT helpers. Keep this
    isolated from GPU tests and bound execution so interpreter regressions
    cannot stall the suite. The reference declares the pairs independently.
    """
    triton = pytest.importorskip("triton")
    if not hasattr(triton, "jit"):
        pytest.skip("A complete Triton runtime is required")
    script = textwrap.dedent(
        """
        import torch
        from rl_engine.reference.reduction.ordered_reduce import ordered_tile_reduce
        from tests.helpers.ordered_reduce_kernels import (
            caller_add, caller_multiply, tile_reduce_probe,
        )
        from tests.helpers.tensor_checks import assert_bitwise

        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(939)
        for width in (1, 8, 16, 64, 1024):
            for group in (1, 8, 32):
                for combine, reference in ((caller_add, torch.add), (caller_multiply, torch.mul)):
                    values = 1 + torch.randn((2, width), generator=generator) * 0.01
                    values[0].fill_(-0.0)
                    expected = ordered_tile_reduce(
                        values, max_group_size=group, combine_fn=reference,
                    )
                    actual = torch.empty((2,), dtype=torch.float32)
                    tile_reduce_probe[(2,)](
                        values, actual, BLOCK_SIZE=width, MAX_GROUP_SIZE=group,
                        COMBINE_FN=combine, num_warps=4, enable_fp_fusion=False,
                    )
                    assert_bitwise(actual, expected, f"interpreter tree: {width}/{group}")

        from rl_engine.backends.shared.triton import reduction as ordered
        from rl_engine.reference import reduction as reference
        from tests.helpers.ordered_reduce_kernels import input_dispatch_probe

        # Check both specializations of the same public function in one kernel.
        # The timeout also bounds this small dispatch matrix.
        for width, length in ((16, 0), (16, 3), (64, 32)):
            values = torch.ones((2, width), dtype=torch.float32)
            values[:, length:] = float("nan")
            for operation in ("sum", "product"):
                outputs = [torch.empty(2) for _ in range(2)]
                input_dispatch_probe[(2,)](
                    values, *outputs, WIDTH=width, LENGTH=length,
                    REDUCE_FN=getattr(ordered, operation), COMPUTE_DTYPE=None,
                    POINTER_BLOCK_SIZE=32, num_warps=4, enable_fp_fusion=False,
                )
                ref = getattr(reference, operation)
                assert_bitwise(outputs[0], ref(values, length=length))
                assert_bitwise(
                    outputs[1], ref(values, length=length, from_memory=True, block_size=32),
                )

        from tests.helpers.ordered_reduce_kernels import (
            boolean_xor, custom_input_probe, custom_partial_probe, custom_merge_probe,
        )
        values = torch.arange(128, dtype=torch.int64).reshape(2, 64)
        values[0, 0] = 2**53 + 1
        for length in (0, 33):
            expected = torch.zeros(2, dtype=torch.int64)
            for column in range(length):
                expected = expected ^ values[:, column]
            outputs = [torch.empty_like(expected) for _ in range(2)]
            custom_input_probe[(2,)](
                values, *outputs, WIDTH=64, LENGTH=length,
                COMBINE_FN=boolean_xor, IDENTITY=0,
            )
            for output in outputs:
                assert_bitwise(output, expected)
            n_tiles = (length + 31) // 32
            partials = torch.empty((2, n_tiles), dtype=torch.int64)
            if n_tiles:
                custom_partial_probe[(2, n_tiles)](
                    values, partials, WIDTH=64, LENGTH=length,
                    COMBINE_FN=boolean_xor, IDENTITY=0,
                )
            initial = torch.tensor([17, 29], dtype=torch.int64)
            custom_merge_probe[(2,)](
                partials, initial, outputs[0], N_TILES=n_tiles,
                COMBINE_FN=boolean_xor, MERGE_UNROLL=3,
            )
            assert_bitwise(outputs[0], expected ^ initial)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[5],
        env={**os.environ, "TRITON_INTERPRET": "1"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("merge_unroll", (1, 2, 3, 4, 8, 16))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("operation", ("sum", "product"))
def test_merge_unroll_preserves_left_fold_and_handles_tails(merge_unroll, dtype, operation):
    """Use order-sensitive partials, runtime seeds and empty/short/uneven merges.

    The oracle folds the given partials directly with PyTorch, independently of
    tile reduction. Cancellation detects reassociation; negative zero detects
    extra additions used to pad a tail. A nonidentity seed must enter only once.
    """
    import triton.language as tl

    from rl_engine.backends.shared.triton.reduction.ordered_reduce import (
        reduce_add,
        reduce_multiply,
    )
    from tests.helpers.ordered_reduce_kernels import ordered_reduce_merge_probe

    kernel_dtype = tl.float32 if dtype == torch.float32 else tl.float64
    if operation == "sum":
        combine, reference_combine = reduce_add, torch.add
        large = 2.0 / torch.finfo(dtype).eps
        patterns = torch.tensor(
            [[large, 1.0, -large, 1.0], [1.0] * 4, [-0.0] * 4, [2.0, 3.0, 5.0, 7.0]],
            dtype=dtype,
        )
        seeds = torch.tensor([0.0, large, -0.0, 7.0], dtype=dtype)
    else:
        combine, reference_combine = reduce_multiply, torch.mul
        patterns = torch.tensor(
            [[2.0, 0.5, 4.0, 0.25], [1.0] * 4, [2.0, 0.5, -2.0, -0.5], [0.5] * 4],
            dtype=dtype,
        )
        seeds = torch.tensor([3.0, -2.0, -0.0, 7.0], dtype=dtype)
    initials = seeds.cuda()

    # Include only-tail, exact groups and groups plus a tail for every factor.
    tile_counts = sorted(
        {0, 1, merge_unroll - 1, merge_unroll, merge_unroll + 1, 2 * merge_unroll + 1}
    )
    for n_tiles in tile_counts:
        cpu_partials = patterns.repeat(1, (n_tiles + 3) // 4)[:, :n_tiles].contiguous()
        expected = seeds.clone()
        for tile in range(n_tiles):
            expected = reference_combine(expected, cpu_partials[:, tile])
        partials = cpu_partials.cuda()
        actual = torch.full((4,), float("nan"), device="cuda", dtype=dtype)
        ordered_reduce_merge_probe[(4,)](
            partials,
            initials,
            actual,
            N_TILES=n_tiles,
            MERGE_UNROLL=merge_unroll,
            COMBINE_FN=combine,
            COMPUTE_DTYPE=kernel_dtype,
            num_warps=4,
            enable_fp_fusion=False,
        )
        assert_bitwise(
            actual.cpu(),
            expected,
            f"{operation}: unroll={merge_unroll}, tiles={n_tiles}",
        )
        assert_bitwise(partials.cpu(), cpu_partials, "merge preserves partials")
    assert_bitwise(initials.cpu(), seeds, "merge preserves runtime seeds")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("n_tiles", (63, 64, 65, 255, 256, 257, 513))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("merge_unroll", (1, 3))
def test_prefetched_merge_preserves_rounding_seed_and_window_boundaries(
    n_tiles, dtype, operation, merge_unroll
):
    """Window reads change neither callback order nor tail/seed arithmetic.

    A separate scalar PyTorch fold supplies the expected bits. Caller-owned
    callbacks, cancellation, negative zero, nonidentity seeds and graph replay
    cover the prefetch threshold, full windows and incomplete final windows.
    """
    import triton.language as tl

    from tests.helpers.ordered_reduce_kernels import (
        caller_add,
        caller_multiply,
        ordered_reduce_merge_probe,
    )

    if operation == "sum":
        combine, reference_combine = caller_add, torch.add
        large = 2.0 / torch.finfo(dtype).eps
        pattern = [[large, 1.0, -large, 1.0], [-0.0] * 4, [1.0] * 4]
        seeds = torch.tensor([0.0, -0.0, large], dtype=dtype)
    else:
        combine, reference_combine = caller_multiply, torch.mul
        pattern = [[2.0, 0.5, 4.0, 0.25], [2.0, 0.5, -2.0, -0.5], [1.0] * 4]
        seeds = torch.tensor([3.0, -0.0, -2.0], dtype=dtype)
    cpu_partials = torch.tensor(pattern, dtype=dtype).repeat(1, (n_tiles + 3) // 4)
    cpu_partials = cpu_partials[:, :n_tiles].contiguous()
    expected = seeds.clone()
    for tile in range(n_tiles):
        expected = reference_combine(expected, cpu_partials[:, tile])
    partials, initials = cpu_partials.cuda(), seeds.cuda()
    actual = torch.empty(3, device="cuda", dtype=dtype)

    def call():
        ordered_reduce_merge_probe[(3,)](
            partials,
            initials,
            actual,
            N_TILES=n_tiles,
            MERGE_UNROLL=merge_unroll,
            COMBINE_FN=combine,
            COMPUTE_DTYPE=tl.float32 if dtype == torch.float32 else tl.float64,
            num_warps=4,
            enable_fp_fusion=False,
        )

    call()
    assert_bitwise(actual.cpu(), expected, "prefetched scalar fold")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    graph.replay()
    graph.replay()
    assert_bitwise(actual.cpu(), expected, "prefetched graph replay")
    assert_bitwise(partials.cpu(), cpu_partials, "prefetch preserves partials")
    assert_bitwise(initials.cpu(), seeds, "prefetch preserves seeds")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("merge_unroll", (0, -1))
def test_merge_rejects_nonpositive_unroll(merge_unroll):
    """Reject invalid loop steps before generating a merge kernel."""
    from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure

    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add
    from tests.helpers.ordered_reduce_kernels import ordered_reduce_merge_probe

    partials = torch.ones((1, 1), device="cuda")
    initials = torch.zeros(1, device="cuda")
    actual = torch.empty_like(initials)
    with pytest.raises(CompilationError) as error_info:
        ordered_reduce_merge_probe[(1,)](
            partials,
            initials,
            actual,
            N_TILES=1,
            MERGE_UNROLL=merge_unroll,
            COMBINE_FN=reduce_add,
            num_warps=4,
        )
    # Nested JIT calls wrap the assertion with their own source locations.
    # Require the actual static assertion, not merely any compilation failure.
    error = error_info.value
    while error.__cause__ is not None:
        error = error.__cause__
    assert isinstance(error, CompileTimeAssertionFailure)
    assert error.error_message == "merge unroll must be a positive integer"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize(
    "dtype,operation",
    [
        (dtype, op)
        for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        for op in ("sum", "product")
    ]
    + [(torch.bool, "xor")],
)
@pytest.mark.parametrize("length", (17, 1025, 1033, 1536))
def test_integer_and_custom_boolean_reductions_across_schedules(dtype, operation, length):
    """Exercise typed seeds, padded tails and partial storage outside floating point."""
    import triton.language as tl

    from rl_engine.backends.shared.triton.reduction.ordered_reduce import (
        reduce_add,
        reduce_multiply,
    )
    from tests.helpers.ordered_reduce_kernels import (
        boolean_xor,
        ordered_reduce_merge_probe,
        ordered_reduce_partial_probe,
        ordered_reduce_serial_probe,
    )

    kernel_dtype = {
        torch.uint8: tl.uint8,
        torch.int8: tl.int8,
        torch.int16: tl.int16,
        torch.int32: tl.int32,
        torch.int64: tl.int64,
        torch.bool: tl.int1,
    }[dtype]
    values = torch.ones((2, length), dtype=dtype, device="cuda")
    initials = torch.tensor([3, 7], dtype=dtype, device="cuda")
    if operation == "xor":
        combine, reference_combine, identity = boolean_xor, torch.bitwise_xor, False
        values[:, 1] = False
        initials = torch.tensor([True, False], device="cuda")
        math_expected = ((values.long().sum(-1) % 2) != 0) ^ initials
    else:
        values[:, :3] = torch.tensor([2, 3, 5], dtype=dtype, device="cuda")
        if dtype == torch.int64:
            values[0, 0] = 2**53 + 1
            initials[1] = 2**53 + 1
        if operation == "sum":
            combine, reference_combine, identity = reduce_add, torch.add, 0
            math_expected = (initials.long() + values.long().sum(-1)).to(dtype)
        else:
            combine, reference_combine, identity = reduce_multiply, torch.mul, 1
            math_expected = (initials.long() * values.long().prod(-1)).to(dtype)
    expected = ordered_reduce(
        values,
        block_size=32,
        **_TREE_CONFIG,
        combine_fn=reference_combine,
        identity=identity,
        initial=initials,
        compute_dtype=dtype,
    )
    assert_bitwise(expected.total, math_expected, "independent integer/boolean oracle")
    common = dict(
        LENGTH=length,
        BLOCK_SIZE=32,
        MAX_GROUP_SIZE=8,
        COMBINE_FN=combine,
        IDENTITY=identity,
        COMPUTE_DTYPE=kernel_dtype,
        num_warps=4,
    )
    actual = torch.empty_like(expected.total)
    for stages in (None, 3):
        ordered_reduce_serial_probe[(2,)](
            values, initials, actual, PIPELINE_STAGES=stages, **common
        )
        assert_bitwise(actual, expected.total, f"{operation}: serial stages={stages}")
    partials = torch.empty_like(expected.partials)
    n_tiles = partials.shape[-1]
    ordered_reduce_partial_probe[(2, n_tiles)](values, partials, **common)
    assert_bitwise(partials, expected.partials, "typed partials")
    for unroll in (1, 3, 4):
        ordered_reduce_merge_probe[(2,)](
            partials,
            initials,
            actual,
            N_TILES=n_tiles,
            MERGE_UNROLL=unroll,
            COMBINE_FN=combine,
            COMPUTE_DTYPE=kernel_dtype,
            num_warps=4,
        )
        assert_bitwise(actual, expected.total, f"{operation}: merge unroll={unroll}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("block_size,length", ((1, 5), (4, 3), (4, 17), (32, 65), (1024, 1025)))
@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("compute_dtype", (*_DTYPES, torch.float64))
def test_operations_with_nonidentity_runtime_seeds(
    dtype, block_size, length, operation, compute_dtype
):
    """Padding identity and final seed must agree across serial/parallel schedules."""
    import triton.language as tl

    kernel_dtype = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
        torch.float64: tl.float64,
    }[compute_dtype]
    from rl_engine.backends.shared.triton.reduction.ordered_reduce import (
        reduce_add,
        reduce_multiply,
    )
    from tests.helpers.ordered_reduce_kernels import (
        ordered_reduce_merge_probe,
        ordered_reduce_partial_probe,
        ordered_reduce_serial_probe,
    )

    columns = torch.arange(length, device="cuda")
    values = (columns % 8 + 2).to(dtype).expand(4, -1).clone()
    if operation == "sum":
        combine, reference_combine, identity = reduce_add, torch.add, 0.0
        initials = torch.tensor([10.0, -3.0, 0.0, 2.0], device="cuda")
    else:
        combine, reference_combine, identity = reduce_multiply, torch.mul, 1.0
        values = torch.where(columns % 2 == 0, 2.0, 0.5).to(dtype).expand(4, -1).clone()
        initials = torch.tensor([3.0, 2.0, 0.5, -2.0], device="cuda")
    expected = ordered_reduce(
        values,
        block_size=block_size,
        combine_fn=reference_combine,
        identity=identity,
        initial=initials,
        max_group_size=8,
        compute_dtype=compute_dtype,
    )
    common = dict(
        LENGTH=length,
        BLOCK_SIZE=block_size,
        MAX_GROUP_SIZE=8,
        COMBINE_FN=combine,
        IDENTITY=identity,
        COMPUTE_DTYPE=kernel_dtype,
        num_warps=4,
        enable_fp_fusion=False,
    )
    initials_before = initials.clone()
    for stages in (None, 3):
        actual = torch.empty_like(expected.total)
        ordered_reduce_serial_probe[(4,)](
            values, initials, actual, PIPELINE_STAGES=stages, **common
        )
        assert_bitwise(actual, expected.total, f"{operation}: serial stages={stages}")
    n_tiles = (length + block_size - 1) // block_size
    partials = torch.empty_like(expected.partials)
    ordered_reduce_partial_probe[(4, n_tiles)](values, partials, **common)
    assert_bitwise(partials, expected.partials, f"{operation}: unseeded partials")
    for unroll in (1, 3, 4):
        actual = torch.empty_like(expected.total)
        ordered_reduce_merge_probe[(4,)](
            partials,
            initials,
            actual,
            N_TILES=n_tiles,
            COMBINE_FN=combine,
            MERGE_UNROLL=unroll,
            COMPUTE_DTYPE=kernel_dtype,
            num_warps=4,
            enable_fp_fusion=False,
        )
        assert_bitwise(actual, expected.total, f"{operation}: merge unroll={unroll}")
    assert_bitwise(initials, initials_before, "caller-owned seed unchanged")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize(
    "compute_dtype,bits",
    (
        (torch.float16, 11),
        (torch.bfloat16, 8),
        (torch.float32, 24),
        (torch.float64, 53),
    ),
)
def test_compute_dtype_rounds_each_add_in_tiles_and_merge(compute_dtype, bits):
    """Both schedules must round at each addition, not just when storing the output."""
    import triton.language as tl

    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add
    from tests.helpers.ordered_reduce_kernels import (
        ordered_reduce_merge_probe,
        ordered_reduce_partial_probe,
        ordered_reduce_serial_probe,
    )

    kernel_dtype = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
        torch.float64: tl.float64,
    }[compute_dtype]
    values = torch.zeros((2, 2049), device="cuda", dtype=torch.float64)
    values[0, :3] = torch.tensor([2**bits, 1, 1], device="cuda", dtype=torch.float64)
    values[1, [0, 1024, 2048]] = values[0, :3]
    expected = torch.full((2,), 2**bits, device="cuda", dtype=compute_dtype)
    initials = torch.zeros(2, device="cuda", dtype=compute_dtype)
    actual = torch.empty_like(expected)
    partials = torch.empty((2, 3), device="cuda", dtype=compute_dtype)
    options = {"num_warps": 4, "enable_fp_fusion": False}
    if torch.version.hip is not None:
        options["allow_flush_denorm"] = False
    tile_options = dict(
        LENGTH=2049,
        BLOCK_SIZE=1024,
        MAX_GROUP_SIZE=8,
        COMBINE_FN=reduce_add,
        IDENTITY=0.0,
        COMPUTE_DTYPE=kernel_dtype,
    )
    for stages in (None, 3):
        ordered_reduce_serial_probe[(2,)](
            values,
            initials,
            actual,
            PIPELINE_STAGES=stages,
            **tile_options,
            **options,
        )
        assert_bitwise(actual, expected, f"per-add rounding, stages={stages}")
    ordered_reduce_partial_probe[(2, 3)](values, partials, **tile_options, **options)
    for unroll in (1, 2, 3, 4, 8):
        ordered_reduce_merge_probe[(2,)](
            partials,
            initials,
            actual,
            N_TILES=3,
            MERGE_UNROLL=unroll,
            COMBINE_FN=reduce_add,
            COMPUTE_DTYPE=kernel_dtype,
            **options,
        )
        assert_bitwise(actual, expected, f"per-add rounding, merge unroll={unroll}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("block_size", (4, 32, 256, 1024))
@pytest.mark.parametrize("max_group_size", (1, 4, 16, 64))
@pytest.mark.parametrize("num_warps", (1, 2, 4, 8))
def test_explicit_group_sizes_match_reference(block_size, max_group_size, num_warps):
    """Group-element selection preserves the same tree across layouts and warp counts.

    Cover singleton groups, one group, groups shorter than a warp, and groups
    spanning multiple lanes. Distinct random values detect cross-group mixing;
    cancellation and negative zero pin the fold's rounding and seed behavior.
    """
    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add
    from tests.helpers.ordered_reduce_kernels import tile_reduce_probe

    generator = torch.Generator().manual_seed(415)
    values = torch.randn((3, block_size), generator=generator).cuda()
    values[0].fill_(-0.0)
    values[1].zero_()
    values[1, :4] = torch.tensor([1e20, 1.0, -1e20, 1.0], device="cuda")
    expected = ordered_tile_reduce(values, max_group_size=max_group_size, combine_fn=torch.add)
    actual = torch.empty_like(expected)
    tile_reduce_probe[(3,)](
        values,
        actual,
        BLOCK_SIZE=block_size,
        MAX_GROUP_SIZE=max_group_size,
        num_warps=num_warps,
        enable_fp_fusion=False,
        COMBINE_FN=reduce_add,
    )
    assert_bitwise(actual, expected, "explicit arithmetic parameters")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("num_warps", (1, 2, 4, 8))
@pytest.mark.parametrize("seed", (415, 438, 939))
@pytest.mark.parametrize("block_size", _TILE_WIDTHS)
def test_production_tile_tree_matches_pytorch_on_identical_fp32_values(num_warps, seed, block_size):
    """Isolate reduction from exp/tanh; changing scheduling cannot change pairs."""
    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add
    from tests.helpers.ordered_reduce_kernels import tile_reduce_probe

    generator = torch.Generator().manual_seed(seed)
    values = torch.randn((16, block_size), generator=generator)
    values *= torch.pow(2.0, torch.randint(-30, 30, values.shape, generator=generator))
    values[0].fill_(-0.0)
    if block_size >= 4:
        values[1].zero_()
        values[1, :4] = torch.tensor([1e20, 1.0, -1e20, 1.0])
    values = values.cuda()
    expected = ordered_tile_reduce(values, **_TREE_CONFIG, combine_fn=torch.add)
    actual = torch.empty(16, device="cuda", dtype=torch.float32)
    tile_reduce_probe[(16,)](
        values,
        actual,
        BLOCK_SIZE=block_size,
        MAX_GROUP_SIZE=_TREE_CONFIG["max_group_size"],
        num_warps=num_warps,
        enable_fp_fusion=False,
        COMBINE_FN=reduce_add,
    )
    assert_bitwise(actual, expected, "isolated production tile tree")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float64))
@pytest.mark.parametrize("block_size", (1, 2, 4, 8, 16, 32, 1024, 2048))
@pytest.mark.parametrize("num_warps", (4, 8))
def test_tile_tree_accepts_other_floating_dtypes(dtype, block_size, num_warps):
    """Exercise direct, sequential and two-operand reduction paths in the caller's dtype."""
    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add
    from tests.helpers.ordered_reduce_kernels import tile_reduce_probe

    generator = torch.Generator().manual_seed(415)
    values = torch.randn((4, block_size), generator=generator).to(device="cuda", dtype=dtype)
    values[0].fill_(-0.0)
    if block_size >= 4:
        # This finite value loses a +1 in each dtype, distinguishing addition trees.
        large = 2.0 / torch.finfo(dtype).eps
        values[1].zero_()
        values[1, :4] = torch.tensor([large, 1.0, -large, 1.0], device="cuda", dtype=dtype)
    expected = ordered_tile_reduce(values, max_group_size=8, combine_fn=torch.add)
    actual = torch.empty(4, device="cuda", dtype=dtype)
    tile_reduce_probe[(4,)](
        values,
        actual,
        BLOCK_SIZE=block_size,
        MAX_GROUP_SIZE=_TREE_CONFIG["max_group_size"],
        num_warps=num_warps,
        enable_fp_fusion=False,
        COMBINE_FN=reduce_add,
    )
    assert_bitwise(actual, expected, "caller-selected dtype and addition tree")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "block_size,length",
    (
        (1, 1),
        (4, 17),
        (16, 48),
        (32, 128),
        (1024, 1025),
        (1024, 1031),
        (1024, 1032),
        (1024, 1033),
        (1024, 4095),
        (1024, 4096),
        (1024, 4097),
        (1024, 4104),
        (1024, 4105),
        (1024, 1536),
        (1024, 1537),
        (1024, 5121),
        (1024, 8192),
        (1024, 8193),
        (1024, 16384),
        (1024, 16385),
        (1024, 17408),
        (1024, 17409),
        (2048, 6147),
        (1024, 263169),
    ),
)
@pytest.mark.parametrize("num_warps", (1, 2, 4))
def test_serial_and_parallel_sums_share_all_rounding_points(dtype, block_size, length, num_warps):
    """Compare partials/totals across tails, pipelining and merging.

    Positive nonassociative values expose a changed addition tree. The source
    buffer already contains the values to reduce; no callback transforms them.
    Four/five and sixteen/seventeen-tile boundaries cover expanded and rolled
    loops. Tails cross group and half-tile boundaries. Parallel producers and
    the independent reference still compute complete padded tiles.
    """
    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add
    from tests.helpers.ordered_reduce_kernels import (
        ordered_reduce_merge_probe,
        ordered_reduce_partial_probe,
        ordered_reduce_serial_probe,
    )

    generator = torch.Generator().manual_seed(415)
    values = torch.randn((4, length), generator=generator).to(device="cuda", dtype=dtype)
    values[0].fill_(1.0)
    values[1].zero_()
    values[1, 0] = 16384.0
    values[1, block_size::block_size] = 2**-10
    values[2].fill_(-0.0)
    expected = ordered_reduce(
        values,
        block_size=block_size,
        max_group_size=8,
        combine_fn=torch.add,
        identity=0.0,
        initial=0.0,
        compute_dtype=torch.float32,
    )
    n_tiles = (length + block_size - 1) // block_size
    math_options = {"enable_fp_fusion": False}
    common = dict(
        LENGTH=length,
        BLOCK_SIZE=block_size,
        MAX_GROUP_SIZE=_TREE_CONFIG["max_group_size"],
        COMBINE_FN=reduce_add,
        IDENTITY=0.0,
        num_warps=num_warps,
        **math_options,
    )
    initials = torch.zeros(4, device="cuda")
    input_before = values.clone()
    for pipeline_stages in (None, 3):
        totals = torch.full_like(expected.total, float("nan"))
        ordered_reduce_serial_probe[(4,)](
            values, initials, totals, PIPELINE_STAGES=pipeline_stages, **common
        )
        assert_bitwise(totals, expected.total, f"serial stages={pipeline_stages}")

    storage = torch.full((4 * n_tiles + 2,), -123.0, device="cuda")
    partials = storage[1:-1].view(4, n_tiles)
    partials.fill_(float("nan"))
    ordered_reduce_partial_probe[(4, n_tiles)](values, partials, **common)
    assert_bitwise(partials, expected.partials, "parallel tile results")
    partials_before = partials.clone()
    for merge_unroll in (1, 2, 3, 4, 8):
        totals = torch.full_like(expected.total, float("nan"))
        ordered_reduce_merge_probe[(4,)](
            partials,
            initials,
            totals,
            N_TILES=n_tiles,
            MERGE_UNROLL=merge_unroll,
            COMBINE_FN=reduce_add,
            num_warps=num_warps,
            **math_options,
        )
        assert_bitwise(totals, expected.total, f"merge unroll={merge_unroll}")
    assert_bitwise(partials, partials_before, "merge preserves saved partials")
    assert_bitwise(values, input_before, "reduction preserves input")
    assert_bitwise(storage[[0, -1]], torch.full((2,), -123.0, device="cuda"), "scratch guards")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("length", (1033, 5121, 16385))
@pytest.mark.parametrize("tile_unroll", (1, 3, 8))
def test_public_tile_unroll_override_preserves_the_declared_tree(operation, length, tile_unroll):
    """Explicit and automatic schedules retain every group, tail and seeded fold."""
    from rl_engine.backends.shared.triton import reduction
    from rl_engine.reference import reduction as reference
    from tests.helpers.ordered_reduce_kernels import tile_unroll_probe

    generator = torch.Generator().manual_seed(1059)
    values = torch.randn((3, length), generator=generator, device="cpu")
    if operation == "sum":
        values[0, :4] = torch.tensor([1e20, 1.0, -1e20, 1.0])
        values[1].fill_(-0.0)
    else:
        values.fill_(1.0)
        values[0, ::2] = 2.0
        values[0, 1::2] = 0.5
        values[1, 0] = -0.0
        values[2, ::2] = -1.0
    expected = getattr(reference, operation)(values, from_memory=True)
    values = values.cuda()
    before = values.clone()
    actual = torch.empty(3, device="cuda")
    for warps in (1, 4):
        for pipeline in (None, 3):
            tile_unroll_probe[(3,)](
                values,
                actual,
                LENGTH=length,
                REDUCE_FN=getattr(reduction, operation),
                TILE_UNROLL=tile_unroll,
                PIPELINE_STAGES=pipeline,
                num_warps=warps,
                enable_fp_fusion=False,
            )
            assert_bitwise(actual.cpu(), expected, f"{operation}: unroll={tile_unroll}")
    assert_bitwise(values, before, "tile-loop expansion preserves inputs")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("tile_unroll", (0, -1))
def test_public_tile_unroll_rejects_nonpositive_factors(tile_unroll):
    from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure

    from rl_engine.backends.shared.triton import reduction
    from tests.helpers.ordered_reduce_kernels import tile_unroll_probe

    values = torch.ones((1, 1033), device="cuda")
    actual = torch.empty(1, device="cuda")
    with pytest.raises(CompilationError) as error_info:
        tile_unroll_probe[(1,)](
            values,
            actual,
            LENGTH=1033,
            REDUCE_FN=reduction.sum,
            TILE_UNROLL=tile_unroll,
            PIPELINE_STAGES=None,
        )

    error = error_info.value
    while error.__cause__ is not None:
        error = error.__cause__
    assert isinstance(error, CompileTimeAssertionFailure)
    assert error.error_message == "tile unroll must be a positive integer"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("operation,identity", (("sum", 0.0), ("product", 1.0)))
@pytest.mark.parametrize(
    "block_size,group_size,length",
    (
        (64, 1, 65),
        (64, 2, 66),
        (64, 8, 7),
        (64, 8, 8),
        (64, 8, 9),
        (64, 8, 15),
        (64, 8, 16),
        (64, 8, 17),
        (64, 8, 31),
        (64, 8, 32),
        (64, 8, 33),
        (64, 1, 79),
        (64, 1, 96),
        (64, 1, 97),
        (64, 8, 264),
        (64, 8, 265),
        (128, 32, 159),
        (128, 32, 160),
        (128, 32, 161),
    ),
)
def test_compact_tail_keeps_padding_tree_and_seed(
    dtype, operation, identity, block_size, group_size, length
):
    """A compact tail must match the full padded tree, including signed-zero combines.

    The oracle materializes every padding group; it does not use the optimized
    compact-tail path. Group and half-tile boundaries select both paths. Nonzero and
    negative-zero seeds must enter once, after the same per-tile computations.
    Caller-owned combiners ensure this is not an addition-only shortcut.
    Padding uses the normal sum/product identities; signed zeros come from
    input tensors and runtime seeds, not Triton negative-zero literals.
    """
    import triton.language as tl

    from tests.helpers.ordered_reduce_kernels import (
        caller_add,
        caller_multiply,
        ordered_reduce_serial_probe,
    )

    values = torch.ones((4, length), dtype=dtype, device="cuda")
    values[1].fill_(-0.0)
    if operation == "sum":
        combine, reference_combine = caller_add, torch.add
        values[0, :3] = torch.tensor([2.0 / torch.finfo(dtype).eps, 1.0, 1.0], device="cuda")
        values[2, 1::2] = -1.0
    else:
        combine, reference_combine = caller_multiply, torch.mul
        values[0, ::2] = 2.0
        values[0, 1::2] = 0.5
        values[2, 1::2] = -1.0
    initials = torch.tensor([3.0, -0.0, -2.0, 0.5], dtype=dtype, device="cuda")
    expected = ordered_reduce(
        values,
        block_size=block_size,
        max_group_size=group_size,
        combine_fn=reference_combine,
        identity=identity,
        initial=initials,
        compute_dtype=dtype,
    ).total
    actual = torch.empty_like(expected)
    for warps in (1, 4):
        for pipeline in (None, 3):
            ordered_reduce_serial_probe[(4,)](
                values,
                initials,
                actual,
                LENGTH=length,
                BLOCK_SIZE=block_size,
                MAX_GROUP_SIZE=group_size,
                COMBINE_FN=combine,
                IDENTITY=identity,
                COMPUTE_DTYPE=tl.float32 if dtype == torch.float32 else tl.float64,
                PIPELINE_STAGES=pipeline,
                num_warps=warps,
                enable_fp_fusion=False,
            )
            assert_bitwise(actual, expected, f"{operation}: tail group, w{warps}/p{pipeline}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("block_size", (8, 1024, 2048))
@pytest.mark.parametrize("num_warps", (1, 2, 4, 8))
def test_eight_lane_extraction_preserves_combiner_order(operation, dtype, block_size, num_warps):
    """Eight-element groups retain sum and product's operand tree across warp counts."""
    from rl_engine.backends.shared.triton.reduction.ordered_reduce import (
        reduce_add,
        reduce_multiply,
    )
    from tests.helpers.ordered_reduce_kernels import tile_reduce_probe

    generator = torch.Generator().manual_seed(428)
    if operation == "sum":
        values = torch.randn((4, block_size), dtype=dtype, generator=generator)
        values[0, :8] = torch.tensor([2**54, 1, -(2**54), 1, 3, -2, 4, -5], dtype=dtype)
        combine, reference = reduce_add, torch.add
    else:
        values = 1.0 + torch.randn((4, block_size), dtype=dtype, generator=generator) * 0.01
        values[0, :8] = torch.tensor([1.1, 0.9, 1.3, 0.7, 1.01, 0.99, 1.2, 0.8], dtype=dtype)
        combine, reference = reduce_multiply, torch.mul
    values[1].fill_(-0.0)
    values = values.cuda()
    expected = ordered_tile_reduce(values, max_group_size=8, combine_fn=reference)
    actual = torch.empty_like(expected)
    tile_reduce_probe[(4,)](
        values,
        actual,
        BLOCK_SIZE=block_size,
        MAX_GROUP_SIZE=8,
        COMBINE_FN=combine,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    assert_bitwise(actual, expected, f"eight-lane extraction: {operation}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("dtype", (*_DTYPES, torch.float64, torch.int16, torch.int64))
@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("num_warps", (1, 2, 4, 8))
def test_two_entry_reduction_matches_custom_combiner_and_reference(dtype, operation, num_warps):
    """Built-in and caller-owned combiners retain the same typed tree.

    Compare raw bits to an independently expressed PyTorch tree. Negative
    zero detects added seeds, cancellation spans both sequential groups and tree
    levels, and int64 values must never pass through floating point.
    """
    from rl_engine.backends.shared.triton.reduction.ordered_reduce import (
        reduce_add,
        reduce_multiply,
    )
    from tests.helpers.ordered_reduce_kernels import caller_add, caller_multiply, tile_reduce_probe

    generator = torch.Generator().manual_seed(939)
    if operation == "sum":
        values = torch.randn((4, 1024), generator=generator).to(dtype)
        combine, custom, reference = reduce_add, caller_add, torch.add
        large = 2.0 / torch.finfo(dtype).eps if dtype.is_floating_point else 8192
        values[1].zero_()
        values[1, [0, 8, 128, 256, 512, 768]] = torch.tensor(
            [large, 1, -large, 1, 1, 1], dtype=dtype
        )
        if dtype == torch.int64:
            values[2, 0] = 2**53 + 1
    else:
        values = (1 + torch.randn((4, 1024), generator=generator) * 0.01).to(dtype)
        combine, custom, reference = reduce_multiply, caller_multiply, torch.mul
        if not dtype.is_floating_point:
            values.fill_(1)
            values[:, 0] = -1
        elif dtype == torch.float64:
            values[2, 0] = 1 + 2**-35
    values[0].fill_(-0.0)
    expected = ordered_tile_reduce(values, max_group_size=8, combine_fn=reference)
    inputs = values.cuda()
    for combiner in (combine, custom):
        actual = torch.empty((4,), dtype=dtype, device="cuda")
        tile_reduce_probe[(4,)](
            inputs,
            actual,
            BLOCK_SIZE=1024,
            MAX_GROUP_SIZE=8,
            COMBINE_FN=combiner,
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
        assert_bitwise(actual.cpu(), expected, f"two-entry pairs: {operation}/{dtype}")
    assert_bitwise(inputs.cpu(), values, "pair reduction preserves inputs")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("dtype", (*_DTYPES, torch.float64))
@pytest.mark.parametrize("block_size", (64, 1024, 2048))
@pytest.mark.parametrize("num_warps", (1, 2, 4, 8))
def test_group_tree_pairs_lower_and_upper_halves_first(dtype, block_size, num_warps):
    """Pin the new pairs with cancellation across halves and an unchanged group fold.

    The first row cancels the large terms before adding the ones, yielding 2.
    In the second row all four terms share a sequential group, yielding 1.
    The previous four-section tree returned 0 for the first row at width 64.
    """
    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add
    from tests.helpers.ordered_reduce_kernels import tile_reduce_probe

    large = 8 / torch.finfo(dtype).eps
    values = torch.zeros((2, block_size), dtype=dtype, device="cuda")
    pattern = torch.tensor([large, 1, -large, 1], dtype=dtype, device="cuda")
    values[0, [0, 8, block_size // 2, block_size // 2 + 8]] = pattern
    values[1, :4] = pattern
    actual = torch.empty((2,), dtype=dtype, device="cuda")
    tile_reduce_probe[(2,)](
        values,
        actual,
        BLOCK_SIZE=block_size,
        MAX_GROUP_SIZE=8,
        COMBINE_FN=reduce_add,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    assert_bitwise(actual, torch.tensor([2, 1], dtype=dtype, device="cuda"))
