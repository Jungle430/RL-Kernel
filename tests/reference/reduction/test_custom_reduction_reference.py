# SPDX-License-Identifier: Apache-2.0
"""Public callback reductions checked against independent scalar arithmetic."""

from functools import reduce as scalar_reduce
from operator import and_, xor

import pytest
import torch

from rl_engine.reference import reduction as ordered
from tests.helpers.tensor_checks import assert_bitwise


def _modular_add(left, right):
    """A caller-owned operation with a neutral zero on residues modulo 257."""
    return (left + right) % 257


_OPERATIONS = (
    pytest.param(torch.bitwise_xor, xor, 0, torch.int64, id="integer-xor"),
    pytest.param(torch.bitwise_and, and_, -1, torch.int64, id="integer-and"),
    pytest.param(_modular_add, _modular_add, 0, torch.int64, id="modular-add"),
    pytest.param(torch.logical_xor, xor, False, torch.bool, id="boolean-xor"),
)


@pytest.mark.parametrize("combine,scalar,identity,dtype", _OPERATIONS)
@pytest.mark.parametrize("width,length", ((16, 0), (16, 11), (64, 33), (1024, 513)))
def test_custom_callback_covers_loaded_memory_and_partial_paths(
    combine, scalar, identity, dtype, width, length
):
    """The callback controls all combines; no sum/product dispatch can pass this matrix."""
    storage = ((torch.arange(4 * width) * 13 + 5) % 257).reshape(2, 2 * width).to(dtype)
    values = storage[:, ::2]
    before = storage.clone()
    expected = torch.tensor(
        [scalar_reduce(scalar, row[:length], identity) for row in values.tolist()], dtype=dtype
    )
    options = dict(combine_fn=combine, identity=identity)
    assert_bitwise(ordered.reduce(values, length=length, **options), expected)
    assert_bitwise(
        ordered.reduce(values, length=length, from_memory=True, block_size=32, **options), expected
    )

    # Producers use the same public helper on loaded tiles. Merge reads fixed
    # slots regardless of the order in which a producer might finish on a GPU.
    n_tiles = (length + 31) // 32
    partials = torch.empty((2, n_tiles), dtype=dtype)
    for tile_index in range(n_tiles - 1, -1, -1):
        start = tile_index * 32
        stop = min(start + 32, length)
        tile = torch.full((2, 32), identity, dtype=dtype)
        tile[:, : stop - start] = values[:, start:stop]
        partials[:, tile_index] = ordered.reduce(tile, **options)
    for unroll in (1, 3):
        assert_bitwise(
            ordered.reduce_partials(
                partials, combine_fn=combine, initial=identity, merge_unroll=unroll
            ),
            expected,
        )
    assert_bitwise(storage, before)


def test_custom_merge_applies_nonidentity_seed_only_once():
    """A nonzero seed distinguishes merging from injecting a value into every tile."""
    partials = torch.tensor([[250, 12, 255, 3], [200, 250, 9, 15]], dtype=torch.int64)
    seeds = torch.tensor([17, 29], dtype=torch.int64)
    before = partials.clone()
    for count in (0, 1, 4):
        expected = torch.tensor(
            [
                scalar_reduce(_modular_add, row[:count], seed)
                for row, seed in zip(partials.tolist(), seeds.tolist(), strict=True)
            ]
        )
        for unroll in (1, 3, 8):
            assert_bitwise(
                ordered.reduce_partials(
                    partials[:, :count], combine_fn=_modular_add, initial=seeds, merge_unroll=unroll
                ),
                expected,
            )
    assert_bitwise(partials, before)


def test_generic_callback_retains_the_declared_loaded_tree_and_memory_seed():
    """Use a caller function plus frozen float results; names must not select arithmetic."""

    def caller_add(left, right):
        return left + right

    values = torch.zeros(64)
    values[[0, 8, 16, 24]] = torch.tensor([1e20, 1, -1e20, 1])
    values[32:] = float("nan")
    options = dict(combine_fn=caller_add, identity=0)
    assert_bitwise(ordered.reduce(values, length=32, **options), torch.tensor(0.0))
    assert_bitwise(
        ordered.reduce(values, length=32, from_memory=True, block_size=32, **options),
        torch.tensor(2.0),
    )
    values = torch.tensor([-0.0])
    assert_bitwise(ordered.reduce(values, **options), values[0])
    assert_bitwise(ordered.reduce(values, from_memory=True, **options), torch.tensor(0.0))


def test_generic_merge_retains_left_association_and_storage_precision():
    """A new tree or implicit dtype conversion would change this frozen result."""

    def caller_add(left, right):
        return left + right

    partials = torch.tensor([[1e20, 1, -1e20, 1]], dtype=torch.float32)
    assert_bitwise(
        ordered.reduce_partials(partials, combine_fn=caller_add, initial=0), torch.tensor([1.0])
    )
    with pytest.raises(ValueError, match="partial storage must match compute dtype"):
        ordered.reduce_partials(
            partials, combine_fn=caller_add, initial=0, compute_dtype=torch.float64
        )
    for unroll in (0, -1):
        with pytest.raises(ValueError, match="merge unroll must be a positive integer"):
            ordered.reduce_partials(partials, combine_fn=caller_add, initial=0, merge_unroll=unroll)


def test_custom_callback_uses_explicit_compute_dtype_without_touching_input():
    values = torch.tensor([2**53 + 1, 4, 8, 0], dtype=torch.int64)
    expected = torch.tensor((2**53 + 1) ^ 4 ^ 8, dtype=torch.int64)
    assert_bitwise(ordered.reduce(values, combine_fn=torch.bitwise_xor, identity=0), expected)
    small = torch.tensor([255, 128, 1, 0], dtype=torch.uint8)
    actual = ordered.reduce(
        small, combine_fn=torch.bitwise_xor, identity=0, compute_dtype=torch.int64
    )
    assert_bitwise(actual, torch.tensor(255 ^ 128 ^ 1, dtype=torch.int64))


@pytest.mark.parametrize("count", (0, 3))
def test_custom_merge_keeps_initial_tensor_outside_autograd(count):
    """A differentiable seed must obey the same reference-only boundary as partials."""
    partials = torch.tensor([[1.0, 2.0, 3.0]])[:, :count]
    initial = torch.tensor([5.0], requires_grad=True)
    with pytest.raises(RuntimeError, match="evaluate under torch.no_grad"):
        ordered.reduce_partials(partials, combine_fn=torch.add, initial=initial)
    with torch.no_grad():
        actual = ordered.reduce_partials(partials, combine_fn=torch.add, initial=initial)
    assert_bitwise(actual, torch.tensor([5.0 if count == 0 else 11.0]))
    # An empty fold may return a view of the seed; no_grad does not clear a
    # view's requires_grad flag, but no backward graph should be constructed.
    assert actual.grad_fn is None
    assert initial.grad is None
