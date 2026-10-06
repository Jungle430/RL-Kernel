# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Public prototype interfaces: native CPU checks and Triton GPU checks."""

import importlib

import pytest
import torch

from rl_engine.kernels.ops.pytorch.norm import NativeFusedAddRMSNormOp

_DTYPES = [
    (torch.float32, torch.float32, torch.float32),
    (torch.float16, torch.float16, torch.float32),
    (torch.bfloat16, torch.bfloat16, torch.float32),
    (torch.bfloat16, torch.float32, torch.bfloat16),
]


@pytest.fixture(scope="module")
def triton_module():
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    triton = pytest.importorskip("triton")
    if not hasattr(triton, "jit"):
        pytest.skip("A working Triton runtime is required; type stubs are insufficient")
    return importlib.import_module("rl_engine.kernels.ops.triton.norm.fused_add_rmsnorm")


@pytest.fixture(params=["native", "triton"])
def implementation(request):
    if request.param == "native":
        return NativeFusedAddRMSNormOp(), "cpu"
    module = request.getfixturevalue("triton_module")
    return module.TritonFusedAddRMSNormOp(), "cuda"


def _rand(shape, seed, *, device="cpu", dtype=torch.float32):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(
        device=device, dtype=dtype
    )


def _inputs(shape, dtypes, device, *, noncontiguous=False):
    inputs = []
    for seed, (tensor_shape, dtype) in enumerate(
        zip((shape, shape, (shape[-1],)), dtypes, strict=True), start=100
    ):
        if noncontiguous:
            padded_shape = (*tensor_shape[:-1], 2 * tensor_shape[-1])
            value = _rand(padded_shape, seed, device=device, dtype=dtype)[..., ::2]
        else:
            value = _rand(tensor_shape, seed, device=device, dtype=dtype)
        inputs.append(value.requires_grad_(True))
    return tuple(inputs)


def _fp64_reference(x, residual, weight, eps):
    # Independent high-precision oracle: no calls into either operator or its backward.
    updated = x + residual
    denominator = (updated.square().mean(dim=-1, keepdim=True) + eps).sqrt()
    return (updated / denominator) * weight, updated


def _backward(outputs, inputs, upstream, branch, **kwargs):
    indices = {"both": (0, 1), "y_only": (0,), "residual_only": (1,)}[branch]
    gradients = torch.autograd.grad(
        tuple(outputs[index] for index in indices),
        inputs,
        grad_outputs=tuple(upstream[index] for index in indices),
        allow_unused=True,
        **kwargs,
    )
    return tuple(
        torch.zeros_like(value) if gradient is None else gradient
        for value, gradient in zip(inputs, gradients, strict=True)
    )


def _assert_close(actual, expected):
    tolerance = {torch.float32: 2e-5, torch.float16: 3e-3, torch.bfloat16: 2e-2}[actual.dtype]
    torch.testing.assert_close(actual, expected.to(actual.dtype), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("shape", [(7,), (3, 7), (2, 3, 2688), (2, 0, 7)])
@pytest.mark.parametrize("dtypes", _DTYPES)
@pytest.mark.parametrize("branch", ["both", "y_only", "residual_only"])
def test_public_forward_backward_against_fp64(implementation, shape, dtypes, branch):
    op, device = implementation
    inputs = _inputs(shape, dtypes, device)
    reference_inputs = tuple(t.detach().double().requires_grad_(True) for t in inputs)
    upstream = tuple(_rand(shape, seed, device=device) for seed in (110, 111))
    actual = op(*inputs, eps=1e-5)
    expected = _fp64_reference(*reference_inputs, eps=1e-5)
    for output, reference in zip(actual, expected, strict=True):
        assert output.shape == shape
        assert output.dtype == torch.float32
        _assert_close(output, reference)
    actual_gradients = _backward(actual, inputs, upstream, branch)
    expected_gradients = _backward(
        expected, reference_inputs, tuple(t.double() for t in upstream), branch
    )
    for value, gradient, reference in zip(
        inputs, actual_gradients, expected_gradients, strict=True
    ):
        assert gradient.shape == value.shape
        assert gradient.dtype == value.dtype
        _assert_close(gradient, reference)
    if branch == "residual_only":
        assert torch.equal(actual_gradients[0], upstream[1].to(inputs[0].dtype))
        assert torch.equal(actual_gradients[1], upstream[1].to(inputs[1].dtype))
        assert torch.equal(actual_gradients[2], torch.zeros_like(inputs[2]))


@pytest.mark.parametrize("dtypes", _DTYPES)
def test_noncontiguous_inputs_and_upstream_gradients(implementation, dtypes):
    op, device = implementation
    shape = (3, 7)
    inputs = _inputs(shape, dtypes, device, noncontiguous=True)
    assert all(not value.is_contiguous() for value in inputs)
    upstream = tuple(_rand((3, 14), seed, device=device)[:, ::2] for seed in (120, 121))
    reference_inputs = tuple(t.detach().double().requires_grad_(True) for t in inputs)
    actual = op(*inputs)
    expected = _fp64_reference(*reference_inputs, eps=1e-5)
    for output, reference in zip(actual, expected, strict=True):
        _assert_close(output, reference)
    for gradient, reference in zip(
        _backward(actual, inputs, upstream, "both"),
        _backward(expected, reference_inputs, tuple(t.double() for t in upstream), "both"),
        strict=True,
    ):
        _assert_close(gradient, reference)


@pytest.mark.parametrize("trainable", [0, 1, 2])
def test_only_one_input_requires_grad(implementation, trainable):
    op, device = implementation
    inputs = _inputs((3, 7), _DTYPES[0], device)
    for index, value in enumerate(inputs):
        value.requires_grad_(index == trainable)
    outputs = op(*inputs)
    (outputs[0].sum() + outputs[1].sum()).backward()
    assert inputs[trainable].grad is not None
    for index, value in enumerate(inputs):
        if index != trainable:
            assert value.grad is None


@pytest.mark.parametrize("eps", [0.0, -1e-5, float("nan"), float("inf")])
def test_invalid_epsilon(implementation, eps):
    op, device = implementation
    with pytest.raises(ValueError, match="eps"):
        op(*_inputs((3, 7), _DTYPES[0], device), eps=eps)


@pytest.mark.parametrize("invalid_input", [0, 1, 2])
def test_unsupported_input_dtype(implementation, invalid_input):
    op, device = implementation
    inputs = list(_inputs((3, 7), _DTYPES[0], device))
    inputs[invalid_input] = inputs[invalid_input].detach().to(torch.int64)
    with pytest.raises(TypeError, match="dtype"):
        op(*inputs)


@pytest.mark.parametrize(
    "shapes,match",
    [
        (((), (), (1,)), "D > 0"),
        (((2, 0), (2, 0), (0,)), "D > 0"),
        (((2, 7), (1, 7), (7,)), "residual"),
        (((2, 7), (2, 7), (1, 7)), "weight"),
        (((2, 7), (2, 7), (8,)), "weight"),
    ],
)
def test_invalid_shapes(implementation, shapes, match):
    op, device = implementation
    inputs = tuple(torch.zeros(shape, device=device) for shape in shapes)
    with pytest.raises(ValueError, match=match):
        op(*inputs)


def test_native_rejects_mismatched_devices():
    x = torch.zeros(2, 7)
    with pytest.raises(ValueError, match="same device"):
        NativeFusedAddRMSNormOp()(x, torch.empty_like(x, device="meta"), torch.ones(7))


@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_repeated_backward_preserves_saved_tensors(triton_module, eps):
    inputs = _inputs((3, 7), _DTYPES[0], "cuda", noncontiguous=True)
    outputs = triton_module.TritonFusedAddRMSNormOp()(*inputs, eps=eps)
    saved = outputs[0].grad_fn.saved_tensors
    before = tuple(value.clone() for value in saved)
    upstream = tuple(_rand((3, 7), seed, device="cuda") for seed in (130, 131))
    first = _backward(outputs, inputs, upstream, "both", retain_graph=True)
    second = _backward(outputs, inputs, upstream, "both")
    for lhs, rhs in zip(first, second, strict=True):
        assert torch.equal(lhs.view(torch.uint8), rhs.view(torch.uint8))
    for lhs, rhs in zip(saved, before, strict=True):
        assert torch.equal(lhs.view(torch.uint8), rhs.view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_train_inference_and_row_invariance(triton_module, dtype):
    op = triton_module.TritonFusedAddRMSNormOp()
    inputs = _inputs((7, 2688), (dtype, dtype, torch.float32), "cuda")
    outputs = op(*inputs)
    for context in (torch.no_grad(), torch.inference_mode()):
        with context:
            inference_outputs = op(*inputs)
        for training, inference in zip(outputs, inference_outputs, strict=True):
            assert not inference.requires_grad
            assert torch.equal(training.view(torch.uint8), inference.view(torch.uint8))

    upstream = tuple(_rand((7, 2688), seed, device="cuda") for seed in (140, 141))
    full_gradients = _backward(outputs, inputs, upstream, "both")
    order = torch.tensor([5, 2, 0], device="cuda")
    subset_inputs = tuple(value[order].detach().requires_grad_(True) for value in inputs[:2])
    subset_inputs += (inputs[2].detach().requires_grad_(True),)
    subset_outputs = op(*subset_inputs)
    subset_gradients = _backward(
        subset_outputs, subset_inputs, tuple(value[order] for value in upstream), "both"
    )
    for full, subset in zip(outputs, subset_outputs, strict=True):
        assert torch.equal(full[order].view(torch.uint8), subset.view(torch.uint8))
    # Weight gradients sum different logical rows; compare only the per-row input gradients.
    for full, subset in zip(full_gradients[:2], subset_gradients[:2], strict=True):
        assert torch.equal(full[order].view(torch.uint8), subset.view(torch.uint8))


@pytest.mark.parametrize("branch", ["grad_y", "grad_updated_residual_output"])
@pytest.mark.parametrize("invalid", ["shape", "dtype", "device"])
def test_backward_rejects_invalid_upstream(triton_module, branch, invalid):
    inputs = _inputs((3, 7), _DTYPES[0], "cuda")
    bad = {
        "shape": torch.zeros(7, device="cuda"),
        "dtype": torch.zeros(3, 7, device="cuda", dtype=torch.int32),
        "device": torch.zeros(3, 7),
    }[invalid]
    upstream = {"grad_y": None, "grad_updated_residual_output": None, branch: bad}
    error = TypeError if invalid == "dtype" else ValueError
    with pytest.raises(error, match=branch):
        triton_module._launch_fused_add_rmsnorm_bwd(*inputs, **upstream)


def test_triton_rejects_cpu_inputs(triton_module):
    with pytest.raises(ValueError, match="device"):
        triton_module.TritonFusedAddRMSNormOp()(*_inputs((3, 7), _DTYPES[0], "cpu"))
