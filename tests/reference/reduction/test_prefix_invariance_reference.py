# SPDX-License-Identifier: Apache-2.0
"""Same logical prefix under different storage and batching, evaluated eagerly."""

import pytest
import torch

from rl_engine.reference.reduction import product as ordered_product
from rl_engine.reference.reduction import sum as ordered_sum
from tests.helpers.ordered_reduce_prefix_cases import prefix_layouts, prefix_storage, prefix_values
from tests.helpers.tensor_checks import assert_bitwise


@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
@pytest.mark.parametrize("block_size", (32, 1024))
@pytest.mark.parametrize("length", (0, 1, 31, 32, 33, 1000, 1001, 1024, 1025, 2049))
def test_same_prefix_is_independent_of_batch_and_storage(operation, dtype, block_size, length):
    """Do not confuse physical row width with the number of effective elements.

    This checks a prefix against itself, not the sum of a shorter prefix plus
    one new element. The reference keeps a fixed tile width and FP32 arithmetic.
    Native numerical-accuracy checks live separately in the reference API tests.
    """
    reduce = ordered_sum if operation == "sum" else ordered_product
    values = prefix_values(length, dtype, operation)
    options = dict(from_memory=True, compute_dtype=torch.float32, block_size=block_size)
    expected = reduce(values, **options)
    for batch_size, width, target_row in prefix_layouts(length, block_size):
        storage = prefix_storage(values, batch_size, width, target_row)
        before = storage.clone()
        actual = reduce(storage, length=length, **options)
        assert_bitwise(actual[target_row], expected, "prefix in a padded batch")
        order = torch.arange(batch_size - 1, -1, -1)
        reordered = reduce(storage[order], length=length, **options)
        assert_bitwise(reordered, actual[order], "batch permutation")
        assert_bitwise(storage, before, "prefix and poisoned suffix preserved")
