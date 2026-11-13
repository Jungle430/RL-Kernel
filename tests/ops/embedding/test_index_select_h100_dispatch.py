# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU dispatch checks plus native workspace / graph regression coverage."""

from types import SimpleNamespace

import pytest
import torch

from rl_engine.backends.cuda.embedding import index_select as native
from tests.ops.embedding.test_index_select import assert_bits, requires_h100, vjp


def test_hardware_properties_use_tensor_ordinal(monkeypatch):
    native._qualified_device.cache_clear()
    calls = []

    def properties(ordinal):
        calls.append(ordinal)
        return SimpleNamespace(
            major=9, minor=0, name="NVIDIA H100 NVL" if ordinal == 1 else "NVIDIA H200"
        )

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    try:
        assert not native._qualified_device(0)
        assert native._qualified_device(1)
        assert native._qualified_device(1)
        assert calls == [0, 1]
    finally:
        native._qualified_device.cache_clear()


def test_dispatch_uses_actual_tensor_and_preserves_fallback(monkeypatch):
    op = native.CudaIndexSelectOp.__new__(native.CudaIndexSelectOp)
    op.h100 = lambda x, index, *, dim: ("cuda", x.device.index, dim)
    op.shared = lambda x, index, *, dim: ("triton", x.device.index, dim)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(native, "_qualified_device", lambda ordinal: ordinal == 1)
    monkeypatch.setattr(native, "_supported_shape", lambda x, index, dim: dim == 1)
    monkeypatch.setattr(native, "_stream_is_capturing", lambda ordinal: False)
    tensor = SimpleNamespace(device=torch.device("cuda:1"))
    assert op(tensor, None, dim=1) == ("cuda", 1, 1)
    monkeypatch.setattr(native, "_stream_is_capturing", lambda ordinal: ordinal == 1)
    assert op(tensor, None, dim=1) == ("triton", 1, 1)
    monkeypatch.setattr(native, "_stream_is_capturing", lambda ordinal: False)
    assert op(tensor, None, dim=0) == ("triton", 1, 0)
    tensor.device = torch.device("cuda:0")
    assert op(tensor, None, dim=1) == ("triton", 0, 1)
    tensor.device = torch.device("cpu")
    assert op(tensor, None, dim=1) == ("triton", None, 1)


def test_shape_limits_do_not_inspect_data():
    index = torch.empty(3, dtype=torch.long, device="meta")
    assert native._supported_shape(torch.empty(2, 9, 7, device="meta"), index, -2)
    assert not native._supported_shape(torch.empty(2**31, device="meta"), index, 0)
    assert not native._supported_shape(torch.empty(2, 9, device="meta"), index, 2)
    max_axis = (native._MAX_WORKSPACE_BYTES // 4 - 2) // 33
    assert native._supported_shape(torch.empty(max_axis, device="meta"), index, 0)
    assert not native._supported_shape(torch.empty(max_axis + 1, device="meta"), index, 0)
    assert not native._supported_shape(torch.empty([1] * 33, device="meta"), index, 0)
    assert not native._supported_shape(torch.empty(65536, 65536, 0, 7, device="meta"), index, 3)


@requires_h100
def test_fp32_subnormals_are_not_flushed():
    """Fast math/FTZ would violate the ordered FP32 contract even without repeats."""
    from rl_engine.reference.embedding.index_select import NativeIndexSelectOp

    x = torch.zeros(1, 17, 7, device="cuda")
    index = torch.arange(257, device="cuda") % 7
    tiny = torch.tensor([2.0**-149, -(2.0**-148), 2.0**-147], device="cuda")
    grad = tiny[torch.arange(257 * 7, device="cuda") % 3].reshape(1, 257, 7)
    expected = vjp(NativeIndexSelectOp(), x, index, grad)[1]
    assert torch.count_nonzero(expected) > 0
    assert_bits(vjp(native.H100IndexSelectOp(), x, index, grad)[1], expected)


@requires_h100
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_workspace_slot_boundary_and_graph_replay(dtype):
    """Identical selected data must agree across 128/32-slot and overflow paths."""
    from rl_engine.reference.embedding.index_select import NativeIndexSelectOp

    generator = torch.Generator().manual_seed(684)
    index_cpu = torch.arange(4).repeat_interleave(129)
    index_cpu = index_cpu[torch.randperm(index_cpu.numel(), generator=generator)]
    index = index_cpu.cuda()
    grad = torch.randn(1, index.numel(), 7, generator=generator).to("cuda", dtype)
    boundary = (native._MAX_WORKSPACE_BYTES // 4 - 1 - index.numel() // 5) // 129
    op = native.H100IndexSelectOp()
    selected = None
    for axis in (17, boundary, boundary + 1):
        x = torch.zeros(1, axis, 7, device="cuda", dtype=dtype)
        expected = vjp(NativeIndexSelectOp(), x, index, grad)[1]
        actual = vjp(op, x, index, grad)[1]
        assert_bits(actual, expected)
        if selected is not None:
            assert_bits(actual[:, :4], selected)
        selected = actual[:, :4].clone()
    # Replay builds and consumes integer grouping on the current nondefault
    # stream. Change upstreams and index contents between replays to rule out stale grouping.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        leaf = x.detach().requires_grad_(True)
        for _ in range(3):
            torch.autograd.grad(op(leaf, index, dim=1), leaf, grad)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = torch.autograd.grad(op(leaf, index, dim=1), leaf, grad)[0]
        for factor in (1.0, -2.0, 0.5):
            grad.mul_(factor)
            index.copy_(index.roll(1))
            graph.replay()
            expected = vjp(NativeIndexSelectOp(), x, index, grad)[1]
            assert_bits(captured, expected)
    torch.cuda.current_stream().wait_stream(stream)


@requires_h100
def test_registry_uses_native_eager_and_shared_capture():
    from rl_engine.ops.autograd.backward_runtime import snapshot_backward_runtime
    from rl_engine.reference.embedding.index_select import NativeIndexSelectOp
    from rl_engine.runtime.registry import KernelRegistry

    op = KernelRegistry().get_op("index_select", device="cuda")
    x = torch.randn(1, 67, 96, device="cuda", requires_grad=True)
    index = torch.arange(33, device="cuda") % 17
    grad = torch.randn(1, index.numel(), 96, device="cuda")
    expected = vjp(NativeIndexSelectOp(), x, index, grad)[1]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.autograd.grad(op(x, index, dim=1), x, grad)
        assert snapshot_backward_runtime()["index_select"]["family"] == "cuda"
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = op(x, index, dim=1)
            captured = torch.autograd.grad(output, x, grad)[0]
        assert snapshot_backward_runtime()["index_select"]["family"] == "triton"
        graph.replay()
        assert_bits(captured, expected)
    torch.cuda.current_stream().wait_stream(stream)


@requires_h100
def test_registry_empty_large_prefix_falls_back():
    """Zero elements do not imply that every intermediate prefix fits int32."""
    from rl_engine.ops.autograd.backward_runtime import snapshot_backward_runtime
    from rl_engine.runtime.registry import KernelRegistry

    op = KernelRegistry().get_op("index_select", device="cuda")
    x = torch.empty(65536, 65536, 0, 7, device="cuda", requires_grad=True)
    index = torch.tensor([6, 0, 6], device="cuda")
    output = op(x, index, dim=3)
    grad = torch.autograd.grad(output, x, torch.empty_like(output))[0]
    assert output.shape == (65536, 65536, 0, 3)
    assert grad.shape == x.shape
    assert snapshot_backward_runtime()["index_select"]["family"] == "triton"
