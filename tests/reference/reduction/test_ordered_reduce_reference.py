# SPDX-License-Identifier: Apache-2.0
"""Fixed-order reduction conformance, independent of model/operator tests."""

import pytest
import torch

from rl_engine.reference.reduction.ordered_reduce import ordered_tile_reduce
from tests.helpers.ordered_reduce_reference import ordered_reduce
from tests.helpers.tensor_checks import assert_bitwise

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_TILE_WIDTHS = tuple(1 << exponent for exponent in range(14))
_TREE_CONFIG = {"max_group_size": 8}
"""Eight-value sequential groups, then one tree over all group totals."""


@pytest.mark.parametrize(
    "compute_dtype,bits",
    ((torch.float16, 11), (torch.bfloat16, 8), (torch.float32, 24), (torch.float64, 53)),
)
def test_compute_dtype_controls_tile_and_merge_rounding(compute_dtype, bits):
    """Independent expected bits distinguish stepwise rounding from an output-only cast."""
    values = torch.zeros((2, 2049), dtype=torch.float64)
    values[0, :3] = torch.tensor([2**bits, 1, 1], dtype=torch.float64)
    values[1, [0, 1024, 2048]] = values[0, :3]
    result = ordered_reduce(
        values,
        block_size=1024,
        **_TREE_CONFIG,
        combine_fn=torch.add,
        identity=0.0,
        initial=0.0,
        compute_dtype=compute_dtype,
    )
    assert result.partials.dtype == result.total.dtype == compute_dtype
    assert_bitwise(result.total, torch.full((2,), 2**bits, dtype=compute_dtype))
    # The fixed path intentionally loses increments; the independent wider sum
    # can distinguish that from the correctly rounded exact sum for these inputs.
    assert result.total[0].item() != 2**bits + 2


def test_fp64_compute_preserves_input_and_initial_bits():
    result = ordered_reduce(
        torch.tensor([1 + 2**-35], dtype=torch.float64),
        block_size=4,
        **_TREE_CONFIG,
        combine_fn=torch.add,
        identity=0.0,
        initial=2**-34,
        compute_dtype=torch.float64,
    )
    assert_bitwise(result.total, torch.tensor(1 + 3 * 2**-35, dtype=torch.float64))


def test_integer_input_and_seed_never_pass_through_floating_point():
    """Both input and Python seed contain bits that even FP64 would lose."""
    initial = 2**53 + 1
    values = torch.tensor([2**53 + 1, 2, 3, 4, 5], dtype=torch.int64)
    result = ordered_reduce(
        values,
        block_size=4,
        **_TREE_CONFIG,
        combine_fn=torch.add,
        identity=0,
        initial=initial,
        compute_dtype=torch.int64,
    )
    assert_bitwise(result.partials, torch.tensor([2**53 + 10, 5], dtype=torch.int64))
    assert_bitwise(result.total, torch.tensor(initial + sum(values.tolist()), dtype=torch.int64))


def test_custom_boolean_combiner_with_padding_and_nonidentity_seed():
    """Boolean XOR demonstrates a reduction outside floating-point sum/product."""
    values = torch.tensor([[True, False, True, False, True], [True, True, False, False, False]])
    result = ordered_reduce(
        values,
        block_size=4,
        **_TREE_CONFIG,
        combine_fn=torch.bitwise_xor,
        identity=False,
        initial=True,
        compute_dtype=torch.bool,
    )
    assert_bitwise(result.partials, torch.tensor([[False, True], [False, False]]))
    assert_bitwise(result.total, torch.tensor([False, True]))


@pytest.mark.parametrize("block_size,length", ((1, 5), (4, 3), (4, 17), (32, 65), (1024, 1025)))
@pytest.mark.parametrize("operation", ("sum", "product"))
def test_operation_identity_and_initial_have_distinct_roles(operation, block_size, length):
    """Independent native checks use exactly representable inputs and results."""
    columns = torch.arange(length)
    values = (columns % 8 + 2).float().expand(4, -1).clone()
    if operation == "sum":
        combine, identity = torch.add, 0.0
        initials = torch.tensor([10.0, -3.0, 0.0, 2.0])
        expected = initials + values.sum(dim=-1)
    else:
        combine, identity = torch.mul, 1.0
        values = torch.where(columns % 2 == 0, 2.0, 0.5).expand(4, -1)
        initials = torch.tensor([3.0, 2.0, 0.5, -2.0])
        # A native FP32 product may group the powers of two before their
        # reciprocals and overflow internally; FP64 keeps this oracle finite.
        expected = (initials.double() * values.double().prod(dim=-1)).float()
    initials_before = initials.clone()
    result = ordered_reduce(
        values,
        block_size=block_size,
        combine_fn=combine,
        identity=identity,
        initial=initials,
        max_group_size=8,
        compute_dtype=torch.float32,
    )
    assert_bitwise(result.total, expected, f"{operation}: padding and one final seed")
    assert_bitwise(initials, initials_before, "caller-owned seed unchanged")
    # Changing the final seed must never change independently produced tiles.
    without_seed = ordered_reduce(
        values,
        block_size=block_size,
        combine_fn=combine,
        identity=identity,
        initial=identity,
        max_group_size=8,
        compute_dtype=torch.float32,
    )
    assert_bitwise(result.partials, without_seed.partials, "partials never include initial")


@pytest.mark.parametrize("block_size", (4, 32, 256))
@pytest.mark.parametrize("max_group_size", (1, 4, 16, 64))
def test_explicit_group_sizes(block_size, max_group_size):
    """Cover groups smaller/larger than the tile without adding identity operands."""
    values = torch.ones((2, block_size), dtype=torch.float32)
    result = ordered_tile_reduce(values, max_group_size=max_group_size, combine_fn=torch.add)
    assert_bitwise(result, torch.full((2,), float(block_size)))
    negative_zero = ordered_tile_reduce(
        torch.full_like(values, -0.0),
        max_group_size=max_group_size,
        combine_fn=torch.add,
    )
    assert_bitwise(negative_zero, torch.full((2,), -0.0), "no extra positive-zero operands")


@pytest.mark.parametrize("max_group_size", (0, -1, 3))
def test_rejects_invalid_group_size(max_group_size):
    with pytest.raises(ValueError, match="maximum group size"):
        ordered_tile_reduce(torch.ones(16), max_group_size=max_group_size, combine_fn=torch.add)


@pytest.mark.parametrize("max_group_size", (1, 8))
@pytest.mark.parametrize(
    "dtype,precision_bits",
    ((torch.float16, 11), (torch.bfloat16, 8), (torch.float32, 24), (torch.float64, 53)),
)
def test_group_totals_follow_one_halving_tree(max_group_size, dtype, precision_bits):
    """Freeze pairs independently of the reference implementation.

    Eight group totals first pair 0+4, 1+5, 2+6, 3+7. The first three rows
    cancel large values before adding the ones; the last row rounds both
    ones away in that first step. Adjacent-first and left-fold trees differ.
    """
    large = 2 ** (precision_bits + 2)
    values = torch.zeros((4, 8 * max_group_size), dtype=dtype)
    for row, positions in enumerate(((0, 1, 4, 5), (0, 1, 2, 3), (0, 2, 4, 6), (0, 4, 1, 5))):
        columns = [group * max_group_size for group in positions]
        values[row, columns] = torch.tensor([large, 1.0, -large, 1.0], dtype=dtype)
    actual = ordered_tile_reduce(values, max_group_size=max_group_size, combine_fn=torch.add)
    assert_bitwise(actual, torch.tensor([2, 2, 2, 0], dtype=dtype), "halving tree")


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("block_size", (1, 2, 4, 16, 1024))
@pytest.mark.parametrize("leading_shape", ((), (2, 3), (0,)))
def test_collector_masks_tails_and_accepts_strided_vectors(dtype, block_size, leading_shape):
    """Input conversion and padding apply to already-computed values."""
    length = 2 * block_size + 1
    storage = torch.full((*leading_shape, length * 2), 2, dtype=dtype)
    values = storage[..., ::2]
    result = ordered_reduce(
        values,
        block_size=block_size,
        **_TREE_CONFIG,
        combine_fn=torch.add,
        identity=0.0,
        initial=0.0,
        compute_dtype=torch.float32,
    )
    assert result.total.dtype == result.partials.dtype == torch.float32
    assert_bitwise(
        result.total,
        torch.full(leading_shape, float(2 * length)),
        "whole-vector sum",
    )
    expected_partials = torch.tensor([2 * block_size, 2 * block_size, 2.0]).expand(
        *leading_shape, 3
    )
    assert_bitwise(result.partials, expected_partials, "padding excluded from sum")


@pytest.mark.parametrize("shape", ((), (0,), (3, 0)))
def test_collector_rejects_missing_or_empty_vector(shape):
    with pytest.raises(ValueError, match="nonempty last dimension"):
        ordered_reduce(
            torch.empty(shape),
            block_size=4,
            **_TREE_CONFIG,
            combine_fn=torch.add,
            identity=0.0,
            initial=0.0,
            compute_dtype=torch.float32,
        )


@pytest.mark.parametrize(
    "positions, expected",
    [
        ((0, 1, 2, 3), 1.0),  # Eight-element sequential fold, not adjacent pairs.
        ((0, 8, 512, 520), 2.0),  # Pair groups 0+64 and 1+65 first.
        ((0, 256, 512, 768), 2.0),  # Pair groups 0+64 and 32+96 first.
    ],
)
def test_fixed_tile_tree_has_explicit_nonassociative_pairs(positions, expected):
    tile = torch.zeros((1, 1024), dtype=torch.float32)
    tile[0, list(positions)] = torch.tensor([1e20, 1.0, -1e20, 1.0])
    assert_bitwise(
        ordered_tile_reduce(tile, **_TREE_CONFIG, combine_fn=torch.add),
        torch.tensor([expected]),
        "tile tree",
    )


@pytest.mark.parametrize(
    "dtype, precision_bits",
    ((torch.float16, 11), (torch.bfloat16, 8), (torch.float32, 24), (torch.float64, 53)),
)
@pytest.mark.parametrize("block_size", (4, 16, 32, 1024, 2048))
def test_fixed_tile_tree_rounds_each_add_in_input_dtype(dtype, precision_bits, block_size):
    """Each +1 rounds away; a wider accumulator would preserve their sum."""
    tile = torch.zeros((1, block_size), dtype=dtype)
    tile[0, :3] = torch.tensor([2**precision_bits, 1, 1], dtype=dtype)
    actual = ordered_tile_reduce(tile, **_TREE_CONFIG, combine_fn=torch.add)
    assert_bitwise(actual, torch.tensor([2**precision_bits], dtype=dtype), "per-add rounding")
    assert actual.item() != 2**precision_bits + 2


@pytest.mark.parametrize("block_size", _TILE_WIDTHS)
@pytest.mark.parametrize("leading_shape", ((), (2, 3), (0,)))
def test_fixed_tile_tree_accepts_power_of_two_widths(block_size, leading_shape):
    """Include strided vectors, arbitrary leading dimensions and empty batches."""
    storage = torch.ones((*leading_shape, block_size * 2), dtype=torch.float32)
    values = storage[..., ::2]
    actual = ordered_tile_reduce(values, **_TREE_CONFIG, combine_fn=torch.add)
    expected = torch.full(leading_shape, float(block_size), dtype=torch.float32)
    assert_bitwise(actual, expected, "power-of-two shape and sum")
    assert_bitwise(
        ordered_tile_reduce(torch.full_like(values, -0.0), **_TREE_CONFIG, combine_fn=torch.add),
        torch.full(leading_shape, -0.0, dtype=torch.float32),
        "no extra positive-zero operand inside a tile",
    )


@pytest.mark.parametrize("block_size", (4, 8, 16))
def test_small_tile_tree_preserves_sequential_groups(block_size):
    """A whole-tile reduction must not silently replace the eight-value folds."""
    values = torch.zeros(block_size, dtype=torch.float32)
    values[:4] = torch.tensor([1e20, 1.0, -1e20, 1.0])
    assert_bitwise(
        ordered_tile_reduce(values, **_TREE_CONFIG, combine_fn=torch.add),
        torch.tensor(1.0),
        "small sequential group",
    )


@pytest.mark.parametrize("block_size", _TILE_WIDTHS[5:])
def test_variable_width_tree_preserves_half_pairing(block_size):
    """The first tree level pairs positions separated by half the tile width."""
    values = torch.zeros(block_size, dtype=torch.float32)
    values[:: block_size // 4] = torch.tensor([1e20, 1.0, -1e20, 1.0])
    assert_bitwise(
        ordered_tile_reduce(values, **_TREE_CONFIG, combine_fn=torch.add),
        torch.tensor(2.0),
        "lower-half/upper-half pairing",
    )


@pytest.mark.parametrize("seed", (415, 438, 939))
@pytest.mark.parametrize("block_size", _TILE_WIDTHS)
@pytest.mark.parametrize("max_group_size", (1, 8, 64))
def test_tile_tree_matches_explicit_pairs(seed, block_size, max_group_size):
    """Cross-check a list of explicit operand pairs across widths, groups and batches."""
    generator = torch.Generator().manual_seed(seed)
    storage = torch.randn((2, 3, 2 * block_size), generator=generator)
    storage *= torch.pow(2.0, torch.randint(-30, 30, storage.shape, generator=generator))
    values = storage[..., ::2]
    group_size = min(block_size, max_group_size)
    n_groups = block_size // group_size
    totals = []
    for group in range(n_groups):
        start = group * group_size
        total = values[..., start]
        for offset in range(1, group_size):
            total = total + values[..., start + offset]
        totals.append(total)
    while len(totals) > 1:
        halfway = len(totals) // 2
        totals = [
            left + right for left, right in zip(totals[:halfway], totals[halfway:], strict=True)
        ]
    expected = totals[0]
    assert_bitwise(
        ordered_tile_reduce(values, max_group_size=max_group_size, combine_fn=torch.add),
        expected,
        "explicit pair sequence",
    )


@pytest.mark.parametrize("block_size", (0, -1, 3, 24, 1023))
def test_reference_rejects_invalid_tile_widths(block_size):
    with pytest.raises(ValueError, match="positive power of two"):
        ordered_reduce(
            torch.ones((1, 7)),
            block_size=block_size,
            **_TREE_CONFIG,
            combine_fn=torch.add,
            identity=0.0,
            initial=0.0,
            compute_dtype=torch.float32,
        )
    if block_size >= 0:
        with pytest.raises(ValueError, match="positive power of two"):
            ordered_tile_reduce(torch.ones((1, block_size)), **_TREE_CONFIG, combine_fn=torch.add)


def test_fixed_tile_reference_requires_array():
    with pytest.raises(ValueError, match="values with at least one dimension"):
        ordered_tile_reduce(torch.tensor(1.0), **_TREE_CONFIG, combine_fn=torch.add)


@pytest.mark.parametrize("block_size", _TILE_WIDTHS)
def test_tile_partials_merge_left_to_right_with_a_zero_initial_value(block_size):
    values = torch.zeros((1, 3 * block_size + 1), dtype=torch.float32)
    values[0, ::block_size] = torch.tensor([1e20, 1.0, -1e20, 1.0])
    result = ordered_reduce(
        values,
        block_size=block_size,
        **_TREE_CONFIG,
        combine_fn=torch.add,
        identity=0.0,
        initial=0.0,
        compute_dtype=torch.float32,
    )
    partials, total = result.partials, result.total
    assert_bitwise(partials, values[:, ::block_size], "partial values including masked tail")
    assert_bitwise(total, torch.tensor([1.0]), "ascending tile merge")
