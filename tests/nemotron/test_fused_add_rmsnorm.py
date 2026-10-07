# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Direct GPU checks for fused add RMSNorm and both weight-gradient reductions."""

import importlib

import pytest
import torch

from rl_engine.kernels.ops.pytorch.norm import NativeFusedAddRMSNormOp

_EPS = 1e-5


@pytest.fixture(scope="module")
def kernels():
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required for the Triton kernel checks")
    triton = pytest.importorskip("triton")
    if not hasattr(triton, "jit"):
        pytest.skip("A working Triton runtime is required; type stubs are insufficient")
    module = importlib.import_module("rl_engine.kernels.ops.triton.norm.fused_add_rmsnorm")
    return triton, module


def _rand(shape, seed, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator).to(device="cuda", dtype=dtype)


def _run_kernels(kernels, x, residual, weight, grad_y, grad_updated_residual_output):
    """Allocate contiguous buffers and launch the core kernels directly."""
    triton, module = kernels
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols
    block_size = triton.next_power_of_2(n_cols)

    y = torch.full_like(x, float("nan"), dtype=torch.float32)
    updated_residual = torch.full_like(x, float("nan"), dtype=torch.float32)
    inverse_rms = torch.full((n_rows,), float("nan"), device=x.device, dtype=torch.float32)
    grad_x = torch.full_like(x, float("nan"))
    grad_residual = torch.full_like(residual, float("nan"))
    grad_weight = torch.full_like(weight, float("nan"))
    grad_weight_per_row = torch.full(
        (n_rows, n_cols), float("nan"), device=x.device, dtype=torch.float32
    )

    if n_rows:
        module._fused_add_rmsnorm_fwd_kernel[(n_rows,)](
            x,
            residual,
            weight,
            y,
            updated_residual,
            inverse_rms,
            n_cols,
            _EPS,
            BLOCK_SIZE=block_size,
            num_warps=4,
            enable_fp_fusion=False,
        )
        module._fused_add_rmsnorm_bwd_kernel[(n_rows,)](
            updated_residual,
            inverse_rms,
            weight,
            grad_y,
            grad_updated_residual_output,
            grad_x,
            grad_residual,
            grad_weight_per_row,
            n_cols,
            BLOCK_SIZE=block_size,
            num_warps=4,
            enable_fp_fusion=False,
        )
    module._fused_add_rmsnorm_bwd_weight_kernel[(triton.cdiv(n_cols, 128),)](
        grad_weight_per_row,
        grad_weight,
        n_rows,
        n_cols,
        BLOCK_SIZE=128,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return y, updated_residual, grad_x, grad_residual, grad_weight, grad_weight_per_row, inverse_rms


def _assert_close(actual, expected):
    if expected.dtype == torch.bfloat16:
        tolerance = 2e-2
    elif expected.dtype == torch.float16:
        tolerance = 3e-3
    else:
        tolerance = 2e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (2, 3, 2688)])
@pytest.mark.parametrize(
    "input_dtype,residual_dtype,weight_dtype",
    [
        (torch.float32, torch.float32, torch.float32),
        (torch.float16, torch.float16, torch.float32),
        (torch.bfloat16, torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.float32, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("branch", ["both", "y_only", "residual_only"])
def test_kernels_match_native_autograd(
    kernels, shape, input_dtype, residual_dtype, weight_dtype, branch
):
    x = _rand(shape, 10, input_dtype).requires_grad_(True)
    residual = _rand(shape, 11, residual_dtype).requires_grad_(True)
    weight = _rand((shape[-1],), 12, weight_dtype).requires_grad_(True)
    grad_y = _rand(shape, 13)
    grad_updated_residual_output = _rand(shape, 14)
    if branch == "y_only":
        grad_updated_residual_output.zero_()
    elif branch == "residual_only":
        grad_y.zero_()

    expected_y, expected_updated_residual = NativeFusedAddRMSNormOp()(x, residual, weight, eps=_EPS)
    expected_gradients = torch.autograd.grad(
        (expected_y, expected_updated_residual),
        (x, residual, weight),
        grad_outputs=(grad_y, grad_updated_residual_output),
    )
    y, updated_residual, grad_x, grad_residual, grad_weight, _, _ = _run_kernels(
        kernels, x, residual, weight, grad_y, grad_updated_residual_output
    )

    _assert_close(y, expected_y)
    assert torch.equal(updated_residual, expected_updated_residual)
    for actual, expected in zip(
        (grad_x, grad_residual, grad_weight), expected_gradients, strict=True
    ):
        _assert_close(actual, expected)
    if branch == "residual_only":
        assert torch.equal(grad_x, grad_updated_residual_output.to(x.dtype))
        assert torch.equal(grad_residual, grad_updated_residual_output.to(residual.dtype))
        assert torch.equal(grad_weight, torch.zeros_like(weight))


@pytest.mark.parametrize("n_rows,n_cols", [(0, 7), (1, 7), (4, 1), (4, 257)])
def test_weight_reduction_matches_ordered_fp32_sum(kernels, n_rows, n_cols):
    triton, module = kernels
    # This cancellation pattern distinguishes the required left-fold from a tree sum.
    values = torch.tensor([1e20, 1.0, -1e20, 1.0], device="cuda", dtype=torch.float32)
    rows = values[:n_rows, None].expand(n_rows, n_cols).contiguous()
    actual = torch.full((n_cols,), float("nan"), device="cuda", dtype=torch.float32)
    module._fused_add_rmsnorm_bwd_weight_kernel[(triton.cdiv(n_cols, 128),)](
        rows,
        actual,
        n_rows,
        n_cols,
        BLOCK_SIZE=128,
        num_warps=4,
        enable_fp_fusion=False,
    )
    expected = torch.zeros_like(actual)
    for row in rows:
        expected = expected + row
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("strategy", ["sequential", "tiled"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "n_rows,n_cols",
    [(0, 129), (1, 1), (1, 7), (31, 127), (32, 128), (33, 129), (65, 2688), (257, 129)],
)
def test_weight_reduction_accuracy_tails_and_repeatability(
    kernels, strategy, dtype, n_rows, n_cols
):
    _, module = kernels
    rows = _rand((n_rows, n_cols), 40)
    before = rows.clone()
    # An output guard catches stores beyond the final partial column block.
    buffer = torch.full((n_cols + 8,), float("nan"), device="cuda", dtype=dtype)
    actual = buffer[:n_cols]
    launch_weight_grad = module._WEIGHT_GRAD_LAUNCHERS[module.RMSNormWeightGradStrategy(strategy)]

    def launch():
        launch_weight_grad(
            grad_weight_per_row=rows,
            grad_weight=actual,
            n_rows=n_rows,
            n_cols=n_cols,
        )

    launch()
    expected = rows.double().sum(dim=0).to(dtype)
    _assert_close(actual, expected)
    first = actual.clone()
    actual.fill_(float("nan"))
    launch()
    assert torch.equal(actual.view(torch.uint8), first.view(torch.uint8))
    assert torch.isnan(buffer[n_cols:]).all()
    assert torch.equal(rows, before)


def test_row_results_are_batch_invariant(kernels):
    shape = (7, 2688)
    x = _rand(shape, 20)
    residual = _rand(shape, 21)
    weight = _rand((shape[-1],), 22)
    grad_y = _rand(shape, 23)
    grad_updated_residual_output = _rand(shape, 24)

    full = _run_kernels(kernels, x, residual, weight, grad_y, grad_updated_residual_output)
    subset = _run_kernels(
        kernels,
        x[2:5],
        residual[2:5],
        weight,
        grad_y[2:5],
        grad_updated_residual_output[2:5],
    )
    # Compare outputs, input gradients, per-row weight contributions, and saved stats.
    # The final weight gradient sums different sets of rows and is not compared.
    for index in (0, 1, 2, 3, 5, 6):
        assert torch.equal(subset[index], full[index][2:5])


def test_zero_residual_sum_has_finite_gradients(kernels):
    shape = (3, 7)
    x = _rand(shape, 30)
    residual = -x
    weight = _rand((shape[-1],), 31)
    grad_y = _rand(shape, 32)
    grad_updated_residual_output = _rand(shape, 33)

    y, updated_residual, grad_x, grad_residual, grad_weight, _, _ = _run_kernels(
        kernels, x, residual, weight, grad_y, grad_updated_residual_output
    )
    expected_input_gradient = (
        grad_y * weight * torch.rsqrt(torch.tensor(_EPS, device="cuda"))
        + grad_updated_residual_output
    )
    assert torch.equal(y, torch.zeros_like(y))
    assert torch.equal(updated_residual, torch.zeros_like(updated_residual))
    assert torch.equal(grad_weight, torch.zeros_like(weight))
    _assert_close(grad_x, expected_input_gradient)
    _assert_close(grad_residual, expected_input_gradient)
