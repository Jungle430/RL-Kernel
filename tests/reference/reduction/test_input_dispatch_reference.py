# SPDX-License-Identifier: Apache-2.0
"""Public input modes and prefix masks, separate from independent accuracy tests."""

import pytest
import torch

from rl_engine.reference import reduction as ordered
from rl_engine.reference.reduction.ordered_reduce import ordered_tile_reduce
from tests.helpers.tensor_checks import assert_bitwise


@pytest.mark.parametrize("operation,identity", (("sum", 0), ("product", 1)))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
@pytest.mark.parametrize("width,length", ((1, 0), (1, 1), (16, 3), (1024, 128), (1024, 1024)))
def test_loaded_prefix_masks_poisoned_suffix_without_mutation(
    operation, identity, dtype, width, length
):
    """Keep the loaded width, and prove ignored NaNs cannot enter the result."""
    storage = torch.full((2, 2 * width), float("nan"), dtype=dtype)
    values = storage[:, ::2]
    values[:, :length] = identity
    if length:
        values[:, 0] = 2
    if length > 1:
        values[:, 1] = -0.5
    before = storage.clone()
    reduce = getattr(ordered, operation)
    combine = torch.add if operation == "sum" else torch.mul
    for compute_dtype in (None, torch.float32):
        resolved_dtype = dtype if compute_dtype is None else compute_dtype
        actual = reduce(values, length=length, compute_dtype=compute_dtype)
        padded = torch.full((2, width), identity, dtype=resolved_dtype)
        padded[:, :length] = values[:, :length].to(resolved_dtype)
        expected = ordered_tile_reduce(
            padded, max_group_size=8, section_count=4, combine_fn=combine
        )
        assert_bitwise(actual, expected)
        # The valid numbers above have an exact answer under any association.
        native = torch.sum if operation == "sum" else torch.prod
        exact = native(values[:, :length].double(), dim=-1).to(resolved_dtype)
        assert_bitwise(actual, exact, "independent exact arithmetic")
    assert_bitwise(storage, before)
    if length == width:
        assert_bitwise(reduce(values), reduce(values, length=length))


def test_loaded_prefix_keeps_width_even_when_pointer_tile_size_is_supplied():
    """Frozen cancellation case: width 64 yields 0, while a 32-wide tree yields 2."""
    values = torch.zeros(64)
    values[[0, 8, 16, 24]] = torch.tensor([1e20, 1, -1e20, 1])
    values[32:] = float("nan")
    assert_bitwise(ordered.sum(values, length=32), torch.tensor(0.0))
    assert_bitwise(ordered.sum(values, length=32, block_size=32), torch.tensor(0.0))
    assert_bitwise(ordered.sum(values[:32]), torch.tensor(2.0))
    assert_bitwise(
        ordered.sum(values, length=32, from_memory=True, block_size=32), torch.tensor(2.0)
    )


@pytest.mark.parametrize("operation", ("sum", "product"))
def test_loaded_mask_accepts_an_arbitrary_subset(operation):
    """A caller-owned mask can select a middle block without a loading flag."""
    values = torch.arange(256, dtype=torch.float32) % 2 + 1
    keep = (torch.arange(256) >= 128) & (torch.arange(256) < 136)
    identity = 0 if operation == "sum" else 1
    masked = torch.where(keep, values, identity)
    expected = torch.tensor(12.0 if operation == "sum" else 16.0)
    assert_bitwise(getattr(ordered, operation)(masked), expected)


@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("from_memory", (False, True))
@pytest.mark.parametrize("length", (-1, 17))
def test_prefix_bounds_are_checked_in_both_modes(operation, from_memory, length):
    with pytest.raises(ValueError, match="length must be between zero and the input width"):
        getattr(ordered, operation)(torch.ones(16), length=length, from_memory=from_memory)


@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("width", (0, 3))
def test_length_does_not_bypass_the_loaded_width_contract(operation, width):
    with pytest.raises(ValueError, match="ordered tile width must be a positive power of two"):
        getattr(ordered, operation)(torch.ones(width), length=0)
    expected = torch.tensor(0.0 if operation == "sum" else 1.0)
    assert_bitwise(
        getattr(ordered, operation)(torch.ones(width), length=0, from_memory=True), expected
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_unmasked_loaded_values_do_not_acquire_the_pointer_seed(dtype):
    """The new name retains signed-zero semantics of each original input path."""
    values = torch.tensor([-0.0], dtype=dtype)
    assert_bitwise(ordered.sum(values), values[0])
    assert_bitwise(ordered.sum(values, length=1), values[0])
    assert_bitwise(ordered.sum(values, length=1, from_memory=True), torch.tensor(0.0, dtype=dtype))


def test_reference_package_exposes_convenience_and_custom_operations():
    assert ordered.__all__ == [
        "sum",
        "product",
        "sum_partials",
        "product_partials",
        "reduce",
        "reduce_partials",
    ]
    for old_name in ("sum_tile", "sum_vector", "product_tile", "product_vector"):
        assert not hasattr(ordered, old_name)
