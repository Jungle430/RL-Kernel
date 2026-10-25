# SPDX-License-Identifier: Apache-2.0
"""Bitwise prefix conformance across batching, padding and execution schedules."""

import pytest
import torch

from rl_engine.reference.reduction import product as ordered_product
from rl_engine.reference.reduction import sum as ordered_sum
from tests.helpers.ordered_reduce_prefix_cases import prefix_layouts, prefix_storage, prefix_values
from tests.helpers.tensor_checks import assert_bitwise

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm is required")


@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("length", (0, 1000, 1001, 1024, 1025, 2049))
def test_same_prefix_across_batch_padding_and_schedules(operation, dtype, length):
    """Compare the same effective values, never two different sequence lengths.

    The fixed prefix is evaluated alone and in larger, differently ordered
    batches with NaN-poisoned suffixes. Serial, pipelined and reversed-assignment
    parallel producers all retain the same tree and FP32 partial storage. This
    is a path-conformance check, not an independent accuracy reference.
    """
    from rl_engine.backends.shared.triton import reduction
    from rl_engine.backends.shared.triton.reduction.ordered_reduce import (
        reduce_add,
        reduce_multiply,
    )
    from tests.helpers.ordered_reduce_kernels import ordered_reduce_merge_probe
    from tests.helpers.ordered_reduce_prefix_kernels import (
        prefix_partials_probe,
        prefix_vector_probe,
    )

    block_size = 1024
    n_tiles = (length + block_size - 1) // block_size
    values = prefix_values(length, dtype, operation)
    reference = ordered_sum if operation == "sum" else ordered_product
    reduce = reduction.sum if operation == "sum" else reduction.product
    combine = reduce_add if operation == "sum" else reduce_multiply
    identity = 0.0 if operation == "sum" else 1.0
    expected = reference(
        values, compute_dtype=torch.float32, block_size=block_size, from_memory=True
    ).cuda()
    options = dict(num_warps=4, enable_fp_fusion=False)
    if torch.version.hip is not None:
        options["allow_flush_denorm"] = False

    for batch_size, width, target_row in prefix_layouts(length, block_size):
        cpu_storage = prefix_storage(values, batch_size, width, target_row)
        all_expected = reference(
            cpu_storage[:, :length], compute_dtype=torch.float32, from_memory=True
        ).cuda()
        storage = cpu_storage.cuda()
        before = storage.clone()
        output_storage = torch.full((batch_size + 2,), -123.0, device="cuda")
        output = output_storage[1:-1]

        for stages in (None, 3):
            output.fill_(float("nan"))
            prefix_vector_probe[(batch_size,)](
                storage,
                output,
                storage.stride(0),
                LENGTH=length,
                BLOCK_SIZE=block_size,
                REDUCE_FN=reduce,
                PIPELINE_STAGES=stages,
                **options,
            )
            assert_bitwise(output, all_expected, f"serial prefix, stages={stages}")
            assert_bitwise(output[target_row], expected, "target independent of batch position")

        partials = torch.empty((batch_size, n_tiles), device="cuda", dtype=torch.float32)
        initials = torch.full((batch_size,), identity, device="cuda")
        for reverse in (False, True):
            partials.fill_(float("nan"))
            if n_tiles:
                prefix_partials_probe[(batch_size, n_tiles)](
                    storage,
                    partials,
                    storage.stride(0),
                    LENGTH=length,
                    BLOCK_SIZE=block_size,
                    COMBINE_FN=combine,
                    IDENTITY=identity,
                    REVERSE_TILES=reverse,
                    **options,
                )
            output.fill_(float("nan"))
            ordered_reduce_merge_probe[(batch_size,)](
                partials,
                initials,
                output,
                N_TILES=n_tiles,
                MERGE_UNROLL=3 if reverse else 1,
                COMBINE_FN=combine,
                **options,
            )
            assert_bitwise(output, all_expected, f"parallel prefix, reversed={reverse}")
            assert_bitwise(output[target_row], expected, "same prefix after parallel merge")

        assert_bitwise(storage, before, "prefix and poisoned padding preserved")
        assert_bitwise(
            output_storage[[0, -1]],
            torch.full((2,), -123.0, device="cuda"),
            "output guards",
        )
