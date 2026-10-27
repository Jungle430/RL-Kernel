# SPDX-License-Identifier: Apache-2.0
"""GPU checks for type dispatch, loaded-prefix masking and public argument errors."""

import pytest
import torch

from rl_engine.reference import reduction as reference
from tests.helpers.tensor_checks import assert_bitwise

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm is required")


@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
@pytest.mark.parametrize("width,length", ((16, 0), (16, 3), (64, 32), (1024, 128), (1024, 1024)))
def test_same_public_function_distinguishes_values_and_pointer(operation, dtype, width, length):
    """Identical length arguments must not select the wrong input representation."""
    from rl_engine.backends.shared.triton import reduction as ordered
    from tests.helpers.ordered_reduce_kernels import input_dispatch_probe

    values = torch.ones((3, width), dtype=dtype)
    if operation == "sum":
        values = (torch.arange(3 * width).reshape(3, width) % 17 - 8).to(dtype)
        if width == 64:
            values[0].zero_()
            large = 2 / torch.finfo(dtype).eps
            values[0, [0, 8, 16, 24]] = torch.tensor([large, 1, -large, 1], dtype=dtype)
    else:
        values[:, ::2] = 2
        values[:, 1::2] = 0.5
    values[:, length:] = float("nan")
    pointer_block = 16
    ref = getattr(reference, operation)
    # Keep arithmetic in input dtype to catch implicit widening; both paths are
    # also exercised with explicit FP32 by the existing embedded-helper matrix.
    expected_loaded = ref(values, length=length, block_size=pointer_block)
    expected_memory = ref(values, length=length, block_size=pointer_block, from_memory=True)
    if operation == "sum" and width == 64:
        # The loaded tree cancels before adding ones; each 16-wide pointer
        # tile rounds its one away before the two partials are merged.
        assert expected_loaded[0].item() == 2
        assert expected_memory[0].item() == 0
    source = values.cuda()
    before = source.clone()
    outputs = [torch.full((5,), -123, device="cuda", dtype=dtype) for _ in range(2)]
    options = dict(num_warps=4, enable_fp_fusion=False)
    if torch.version.hip is not None:
        options["allow_flush_denorm"] = False
    assert ordered.__all__ == [
        "sum",
        "product",
        "sum_partials",
        "product_partials",
        "reduce",
        "reduce_partials",
    ]
    input_dispatch_probe[(3,)](
        source,
        outputs[0][1:-1],
        outputs[1][1:-1],
        WIDTH=width,
        LENGTH=length,
        REDUCE_FN=getattr(ordered, operation),
        COMPUTE_DTYPE=None,
        POINTER_BLOCK_SIZE=pointer_block,
        **options,
    )
    for output, expected in zip(outputs, (expected_loaded, expected_memory), strict=True):
        assert_bitwise(output[1:-1], expected.cuda())
        assert_bitwise(output[[0, -1]], torch.full((2,), -123, device="cuda", dtype=dtype))
    assert_bitwise(source, before, "including the NaN suffix")


@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize(
    "mode,length,message",
    (
        ("pointer", None, "pointer input requires length"),
        ("pointer", -1, "vector length must be nonnegative"),
        ("loaded", -1, "loaded length must be between zero and the vector width"),
        ("loaded", 17, "loaded length must be between zero and the vector width"),
        ("pointer_vector", 16, "input must be a scalar pointer"),
    ),
)
def test_public_dispatch_rejects_invalid_arguments_before_launch(operation, mode, length, message):
    from triton.compiler.errors import CompilationError

    from rl_engine.backends.shared.triton import reduction as ordered
    from tests.helpers.ordered_reduce_kernels import invalid_input_probe

    values = torch.ones(16, device="cuda")
    output = torch.empty((), device="cuda")
    with pytest.raises(CompilationError, match=message):
        invalid_input_probe[(1,)](
            values, output, MODE=mode, LENGTH=length, REDUCE_FN=getattr(ordered, operation)
        )
