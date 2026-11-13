# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared index selection: copies, ordered VJP, accuracy and supported invariances."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.reference.embedding.index_select import NativeIndexSelectOp

DTYPES = (torch.float32, torch.float16, torch.bfloat16)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA",
)


def assert_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def vjp(op, x, index, grad, *, dim=1):
    leaf = x.detach().requires_grad_(True)
    output = op(leaf, index, dim=dim)
    return output.detach(), torch.autograd.grad(output, leaf, grad)[0]


class TestReference:
    @pytest.mark.parametrize(
        "shape,count",
        [((2, 0, 32), 0), ((2, 5, 96), 0), ((0, 5, 32), 3), ((2, 5, 0), 3)],
    )
    def test_empty(self, shape, count):
        x = torch.zeros(shape)
        index = torch.zeros(count, dtype=torch.int64)
        grad = torch.ones(shape[0], count, shape[2])
        output, dx = vjp(NativeIndexSelectOp(), x, index, grad)
        assert output.shape == grad.shape
        assert_bits(dx, torch.zeros_like(x))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_duplicate_contract(self, dtype):
        x = torch.randn(2, 5, 3, dtype=dtype)
        index = torch.tensor([4, 1, 4, 0, 4], dtype=torch.int32)
        grad = torch.arange(30).reshape(2, 5, 3).to(dtype)
        output, dx = vjp(NativeIndexSelectOp(), x, index, grad)
        assert_bits(output, x.index_select(1, index))
        expected = torch.zeros_like(x)
        expected[:, 0] = grad[:, 3]
        expected[:, 1] = grad[:, 1]
        expected[:, 4] = (grad[:, 0].float() + grad[:, 2].float() + grad[:, 4].float()).to(dtype)
        assert_bits(dx, expected)

    def test_fp32_rounding_order_is_observable(self):
        x = torch.zeros(1, 1, 1)
        index = torch.zeros(3, dtype=torch.int64)
        grad = torch.tensor([2**24, 1, -(2**24)], dtype=torch.float32).view(1, 3, 1)
        op = NativeIndexSelectOp()
        assert vjp(op, x, index, grad)[1].item() == 0
        # FP64 accuracy oracle deliberately differs on adversarial cancellation.
        assert vjp(op.forward_fp32, x, index, grad)[1].item() == 1
        assert vjp(op, x, index, grad[:, [0, 2, 1]])[1].item() == 1

    @pytest.mark.parametrize(
        "x,index,error",
        [
            (torch.tensor(0.0), torch.tensor([0]), ValueError),
            (torch.zeros(1, 3, 2).double(), torch.tensor([0]), TypeError),
            (torch.zeros(1, 3, 2), torch.tensor([[0]]), ValueError),
            (torch.zeros(1, 3, 2), torch.tensor([0.0]), TypeError),
            (torch.zeros(1, 3, 2), torch.tensor([True]), TypeError),
            (torch.zeros(1, 3, 2), torch.tensor([-1]), IndexError),
            (torch.zeros(1, 3, 2), torch.tensor([3]), IndexError),
            (torch.zeros(1, 0, 2), torch.tensor([0]), IndexError),
        ],
    )
    def test_invalid_inputs(self, x, index, error):
        with pytest.raises(error):
            NativeIndexSelectOp()(x, index, dim=1)

    def test_dim_is_required_and_keyword_only(self):
        op = NativeIndexSelectOp()
        x, index = torch.zeros(2, 3), torch.tensor([0])
        with pytest.raises(TypeError, match="dim"):
            op(x, index)
        with pytest.raises(TypeError):
            op(x, index, 0)


@pytest.mark.parametrize("backend", ["pytorch", pytest.param("triton", marks=requires_cuda)])
class TestSharedAxes:
    @staticmethod
    def operator(backend):
        if backend == "pytorch":
            return NativeIndexSelectOp(), torch.device("cpu")
        from rl_engine.backends.shared.triton.embedding.index_select import (
            TritonIndexSelectOp,
        )

        return TritonIndexSelectOp(), torch.device("cuda")

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize(
        "shape,dim",
        [
            ((7,), 0),
            ((7, 9), 0),
            ((2, 7), -1),
            ((2, 7, 9), 1),
            ((7, 3, 9), 0),
            ((2, 3, 7, 9), 2),
            ((7, 2, 3, 9), -4),
            ((2, 3, 7), -1),
        ],
    )
    def test_axes_strides_ordered_vjp_and_accuracy(self, backend, dtype, shape, dim):
        """H3, BAGEL and arbitrary axes use the same bitwise and accuracy contracts."""
        op, device = self.operator(backend)
        generator = torch.Generator(device=device).manual_seed(680)
        # Reverse physical axes and slice with a storage offset: neither prefix
        # nor suffix dimensions can be assumed contiguous by the kernel.
        storage_shape = list(reversed(shape))
        storage_shape[-1] *= 2
        x = torch.randn(storage_shape, device=device, dtype=dtype, generator=generator)
        x = x[..., 1::2].permute(tuple(reversed(range(len(shape)))))
        index = torch.tensor([6, 0, 1, 0, 6, 0, 0, 0], device=device, dtype=torch.int32)[::2]
        output_shape = list(shape)
        output_shape[dim] = index.numel()
        grad = torch.randn(
            tuple(reversed(output_shape)),
            device=device,
            dtype=dtype,
            generator=generator,
        ).permute(tuple(reversed(range(len(shape)))))
        output, dx = vjp(op, x, index, grad, dim=dim)
        assert output.is_contiguous() and dx.dtype == x.dtype
        assert_bits(output, torch.index_select(x, dim, index))
        expected_dx = torch.zeros(shape, dtype=torch.float32)
        for position, source in enumerate(index.cpu().tolist()):
            expected_dx.select(dim, source).add_(grad.cpu().float().select(dim, position))
        assert_bits(dx.cpu(), expected_dx.to(dtype))
        # Bitwise order and mathematical accuracy are separate assertions.
        gold = vjp(NativeIndexSelectOp().forward_fp32, x.float(), index, grad.float(), dim=dim)[1]
        tolerance = resolve_tolerance(
            load_contract(),
            judgment="gradient_accuracy",
            op_class="reduction",
            dtype=dtype,
        )
        torch.testing.assert_close(dx.float(), gold, atol=tolerance.atol, rtol=tolerance.rtol)
        with torch.inference_mode():
            assert_bits(op(x, index, dim=dim), output)
        assert_bits(vjp(op, x, index, grad, dim=dim)[1], dx)

    @pytest.mark.parametrize(
        "shape,dim,count",
        [
            ((0,), 0, 0),
            ((7,), -1, 0),
            ((0, 7), 1, 2),
            ((7, 0), 0, 2),
            ((2, 0, 3), 1, 0),
        ],
    )
    def test_empty_axes(self, backend, shape, dim, count):
        op, device = self.operator(backend)
        x = torch.zeros(shape, device=device)
        index = torch.zeros(count, device=device, dtype=torch.int64)
        expected = x.index_select(dim, index)
        output, dx = vjp(op, x, index, torch.ones_like(expected), dim=dim)
        assert_bits(output, expected)
        assert_bits(dx, torch.zeros_like(x))

    @pytest.mark.parametrize("dim", [-3, 2])
    def test_invalid_dim(self, backend, dim):
        op, device = self.operator(backend)
        with pytest.raises(IndexError, match="dim must be"):
            op(
                torch.zeros(2, 7, device=device),
                torch.tensor([0], device=device),
                dim=dim,
            )

    @pytest.mark.parametrize("dim", [0, -1])
    def test_cancellation_uses_original_occurrence_order(self, backend, dim):
        op, device = self.operator(backend)
        x = torch.zeros(1, 1, device=device)
        index = torch.zeros(3, device=device, dtype=torch.int64)
        grad = torch.tensor([2**24, 1, -(2**24)], device=device, dtype=torch.float32)
        grad = grad[:, None] if dim == 0 else grad[None, :]
        assert vjp(op, x, index, grad, dim=dim)[1].item() == 0
        assert vjp(NativeIndexSelectOp().forward_fp32, x, index, grad, dim=dim)[1].item() == 1


@requires_cuda
@pytest.mark.cuda_only
class TestCuda:
    @staticmethod
    def op():
        from rl_engine.backends.shared.triton.embedding.index_select import (
            TritonIndexSelectOp,
        )

        return TritonIndexSelectOp()

    @pytest.mark.parametrize(
        "count,axis_size,index_dtype",
        [
            (1, 7, torch.int32),
            (33, 37, torch.int64),
            (2048, 67, torch.int64),
            (2049, 67, torch.int32),
            (64, (2**31 - 1) // 64, torch.int64),
            (64, (2**31 - 1) // 64 + 1, torch.int64),
            (64, 2**35, torch.int64),
        ],
    )
    def test_stable_sort_preserves_occurrences_and_large_indices(
        self, count, axis_size, index_dtype
    ):
        """Exercise small-sort tails, its size boundary, and packed-key overflow."""
        from rl_engine.backends.shared.triton.embedding.index_select import (
            _sort_index_occurrences,
        )

        choices = torch.tensor(
            [0, axis_size - 1, 0, 0, 0, axis_size // 2, 0, axis_size - 1],
            dtype=index_dtype,
        )
        # Include both an offset and stride; keys need not fit int32 on fallback.
        storage = choices.repeat((2 * count + 8) // 8 + 1).to("cuda")
        index = storage[1 : 2 * count + 1 : 2]
        actual_values, actual_order = _sort_index_occurrences(index, axis_size)
        expected_values, expected_order = torch.sort(index, stable=True)
        assert torch.equal(actual_values, expected_values)
        assert torch.equal(actual_order, expected_order)
        assert actual_values.dtype == index_dtype and actual_order.dtype == torch.int64

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("width", [1, 7, 257])
    def test_short_and_long_duplicate_segments_keep_order(self, dtype, width):
        """Mixed segment lengths cross loop, prefetch, row-block and outer tails."""
        generator = torch.Generator().manual_seed(680)
        counts = torch.tensor([1, 3, 4, 5, 16, 17, 18, 33, 128, 129, 256, 257, 2])
        index_cpu = torch.repeat_interleave(torch.arange(len(counts)), counts)
        index_cpu = index_cpu[torch.randperm(index_cpu.numel(), generator=generator)]
        index = index_cpu.to("cuda")
        x = torch.zeros(3, len(counts), width, device="cuda", dtype=dtype)
        grad_cpu = torch.randn(3, width, index.numel(), generator=generator).to(dtype)
        grad_cpu = grad_cpu.transpose(1, 2)
        grad_cpu[:, index_cpu == 0, :] = -0.0
        if dtype == torch.float32:
            occurrences = torch.where(index_cpu == 2)[0]
            grad_cpu[:, occurrences, 0] = 0
            grad_cpu[:, occurrences[:3], 0] = torch.tensor([2**24, 1, -(2**24)], dtype=dtype)
        grad = grad_cpu.to("cuda")
        output, dx = vjp(self.op(), x, index, grad)
        expected = torch.zeros(x.shape, dtype=torch.float32)
        for position, source in enumerate(index_cpu.tolist()):
            expected[:, source].add_(grad_cpu[:, position].float())
        assert_bits(dx.cpu(), expected.to(dtype))
        assert_bits(dx[:, 0], torch.zeros_like(dx[:, 0]))
        assert_bits(vjp(self.op(), x, index, grad)[1], dx)
        for batch in range(x.shape[0]):
            single_y, single_dx = vjp(
                self.op(), x[batch : batch + 1], index, grad[batch : batch + 1]
            )
            assert_bits(single_y, output[batch : batch + 1])
            assert_bits(single_dx, dx[batch : batch + 1])

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("n_selected", [64, 1024, 1025])
    def test_direct_scan_boundary_keeps_long_duplicate_order(self, dtype, n_selected):
        """Long duplicate chains retain their bits across direct/bucket dispatch."""
        generator = torch.Generator().manual_seed(683)
        index_cpu = torch.arange(n_selected) % 5
        index_cpu = index_cpu[torch.randperm(n_selected, generator=generator)]
        index = index_cpu.to("cuda")
        x = torch.zeros(2, 2 * n_selected + 1, 7, device="cuda", dtype=dtype)
        upstream = torch.randn(2, 7, n_selected, generator=generator).to(dtype)
        upstream = upstream.transpose(1, 2)
        upstream[:, index_cpu == 0, :] = -0.0
        if dtype == torch.float32:
            positions = torch.where(index_cpu == 1)[0]
            upstream[:, positions, 0] = 0
            upstream[:, positions[:3], 0] = torch.tensor([2**24, 1, -(2**24)], dtype=dtype)
        grad = upstream.to("cuda")
        expected = vjp(NativeIndexSelectOp(), x, index, grad)[1]
        for _ in range(2):
            assert_bits(vjp(self.op(), x, index, grad)[1], expected)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_bounded_metadata_keeps_gradient_order(self, dtype, monkeypatch):
        """Four/128 slots and stable-sort fallback must produce identical bits."""
        import rl_engine.backends.shared.triton.embedding.index_select as implementation

        generator = torch.Generator().manual_seed(682)
        counts = torch.tensor([1, 4, 5, 16, 17, 128, 129, 257])
        indices = torch.repeat_interleave(torch.arange(len(counts)), counts)
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        storage = torch.empty(indices.numel() * 2, dtype=torch.int64, device="cuda")
        storage[::2] = indices.to("cuda")
        index = storage[::2]
        x = torch.zeros(2, 32, 257, device="cuda", dtype=dtype)
        upstream = torch.randn(2, 257, indices.numel(), generator=generator).to(dtype)
        grad = upstream.transpose(1, 2).to("cuda")
        expected = vjp(NativeIndexSelectOp(), x, index, grad)[1]
        minimum_bytes = 4 * (6 * x.shape[1] + 1 + max(1, index.numel() // 5) + index.numel())
        for budget, slots in ((128 * 1024 * 1024, 128), (minimum_bytes, 4), (0, 0)):
            monkeypatch.setattr(implementation, "_MAX_BUCKET_BYTES", budget)
            assert implementation._dense_bucket_slots(x.shape[1], index.numel()) == slots
            assert_bits(vjp(self.op(), x, index, grad)[1], expected)

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize(
        "width,index_dtype",
        [(32, torch.int32), (96, torch.int64), (7, torch.int64), (513, torch.int32)],
    )
    def test_strides_repeats_accuracy(self, dtype, width, index_dtype):
        torch.manual_seed(420)
        x = torch.randn(2, 67, 2 * width, device="cuda", dtype=dtype)[..., ::2]
        index = torch.randint(0, 67, (258,), device="cuda", dtype=index_dtype)[::2]
        grad = torch.randn(2, width, 129, device="cuda", dtype=dtype).transpose(1, 2)
        output, dx = vjp(self.op(), x, index, grad)
        assert output.is_contiguous()
        assert_bits(output, x.index_select(1, index))
        ordered = vjp(NativeIndexSelectOp(), x, index, grad)[1]
        assert_bits(dx, ordered)
        gold = vjp(NativeIndexSelectOp().forward_fp32, x.float(), index, grad.float())[1]
        tolerance = resolve_tolerance(
            load_contract(),
            judgment="gradient_accuracy",
            op_class="reduction",
            dtype=dtype,
        )
        torch.testing.assert_close(dx.float(), gold, atol=tolerance.atol, rtol=tolerance.rtol)
        for _ in range(3):
            assert_bits(vjp(self.op(), x, index, grad)[1], dx)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_grouping_paths_preserve_identical_occurrence_order(self, dtype):
        """Changing the unused axis extent must not change a selected gradient."""
        generator = torch.Generator().manual_seed(681)
        index_cpu = torch.randint(0, 64, (64,), generator=generator)
        index = index_cpu.to("cuda")
        grad_cpu = torch.randn(2, 64, 129, generator=generator).to(dtype)
        grad = grad_cpu.to("cuda")
        expected = torch.zeros(2, 64, 129, dtype=torch.float32)
        for position, source in enumerate(index_cpu.tolist()):
            expected[:, source].add_(grad_cpu[:, position].float())
        # Same data through direct-small, dense buckets, and sparse stable sort.
        for axis_size in (128, 300, 1025):
            x = torch.zeros(2, axis_size, 129, device="cuda", dtype=dtype)
            _, dx = vjp(self.op(), x, index, grad)
            assert_bits(dx[:, :64].cpu(), expected.to(dtype))
            assert_bits(dx[:, 64:], torch.zeros_like(dx[:, 64:]))

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_many_overflow_groups_keep_original_order(self, dtype):
        """Queue capacity, persistent work, and unordered insertion are observable."""
        generator = torch.Generator().manual_seed(682)
        index_cpu = torch.arange(257).repeat_interleave(5)
        index_cpu = index_cpu[torch.randperm(index_cpu.numel(), generator=generator)]
        index_storage = torch.zeros(index_cpu.numel() * 2, device="cuda", dtype=torch.int64)
        index_storage[1::2] = index_cpu.to("cuda")
        index = index_storage[1::2]
        grad_cpu = torch.randn(2, 257, index.numel(), generator=generator).transpose(1, 2).to(dtype)
        if dtype == torch.float32:
            positions = torch.where(index_cpu == 256)[0]
            grad_cpu[:, positions, 0] = torch.tensor([2**24, 1, -(2**24), -0.0, 1])
        expected = torch.zeros(2, 257, 257, dtype=torch.float32)
        for position, source in enumerate(index_cpu.tolist()):
            expected[:, source].add_(grad_cpu[:, position].float())
        x = torch.zeros(2, 257, 257, device="cuda", dtype=dtype)
        grad = grad_cpu.to("cuda")
        for _ in range(3):
            output, dx = vjp(self.op(), x, index, grad)
            assert_bits(output, x.index_select(1, index))
            assert_bits(dx.cpu(), expected.to(dtype))

    @pytest.mark.parametrize(
        "shape,count",
        [
            ((2, 0, 32), 0),
            ((2, 5, 96), 0),
            ((0, 5, 32), 3),
            ((2, 5, 0), 3),
            ((1, 1, 1), 1),
        ],
    )
    def test_empty_and_boundary(self, shape, count):
        x = torch.empty(shape, device="cuda")
        index = torch.zeros(count, device="cuda", dtype=torch.int64)
        grad = torch.ones(shape[0], count, shape[2], device="cuda")
        output, dx = vjp(self.op(), x, index, grad)
        assert_bits(output, x.index_select(1, index))
        assert_bits(dx, vjp(NativeIndexSelectOp(), x, index, grad)[1])

    @pytest.mark.parametrize("width", [32, 96])
    @pytest.mark.parametrize("sequence", [38734, 110750, 367986])
    def test_realistic_h3_unique_selection(self, width, sequence):
        # Full mixed-row counts from the merged output-head qualification.
        x = torch.randn(1, sequence, width, device="cuda")
        index = torch.arange(sequence - 1, -1, -3, device="cuda")
        grad = torch.randn(1, index.numel(), width, device="cuda")
        output, dx = vjp(self.op(), x, index, grad)
        assert_bits(output, x.index_select(1, index))
        expected = torch.zeros_like(x)
        expected.index_copy_(1, index, grad)
        assert_bits(dx, expected)

    def test_batch_train_infer_checkpoint_and_row_relabeling(self):
        from torch.utils.checkpoint import checkpoint

        x = torch.randn(3, 73, 96, device="cuda")
        index = torch.tensor([72, 1, 0, 72, 2, 1], device="cuda")
        grad = torch.randn(3, 6, 96, device="cuda")
        op = self.op()
        output, dx = vjp(op, x, index, grad)
        with torch.inference_mode():
            assert_bits(op(x, index, dim=1), output)
        for batch in range(3):
            single_y, single_dx = vjp(op, x[batch : batch + 1], index, grad[batch : batch + 1])
            assert_bits(single_y, output[batch : batch + 1])
            assert_bits(single_dx, dx[batch : batch + 1])
        permutation = torch.tensor([2, 0, 1], device="cuda")
        py, pdx = vjp(op, x[permutation], index, grad[permutation])
        assert_bits(py, output[permutation])
        assert_bits(pdx, dx[permutation])
        # Relabel input rows while preserving duplicate occurrence order.
        rows = torch.randperm(73, device="cuda")
        ry, rdx = vjp(op, x[:, rows], torch.argsort(rows)[index], grad)
        assert_bits(ry, output)
        assert_bits(rdx, dx[:, rows])
        cy, cdx = vjp(
            lambda a, i, *, dim: checkpoint(op, a, i, dim=dim, use_reentrant=False),
            x,
            index,
            grad,
        )
        assert_bits(cy, output)
        assert_bits(cdx, dx)
        changed = x.clone()
        changed[0] *= 123
        assert_bits(op(changed, index, dim=1)[1:], output[1:])

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_special_value_copy_and_high_multiplicity(self, dtype):
        x = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan")], device="cuda")
        x = x.view(1, 5, 1)
        index = torch.tensor([4, 1, 3, 0, 2], device="cuda")
        assert_bits(self.op()(x, index, dim=1), x.index_select(1, index))
        x = torch.zeros(2, 3, 32, device="cuda", dtype=dtype)
        index = torch.ones(1025, device="cuda", dtype=torch.int64)
        grad = torch.randn(2, 1025, 32, device="cuda", dtype=dtype)
        assert_bits(
            vjp(self.op(), x, index, grad)[1],
            vjp(NativeIndexSelectOp(), x, index, grad)[1],
        )

    def test_expanded_input_and_gradient(self):
        x = torch.randn(1, 1, 7, device="cuda").expand(2, 9, 7)
        index = torch.tensor([8, 0, 8], device="cuda")
        grad = torch.ones(1, 1, 1, device="cuda").expand(2, 3, 7)
        actual = vjp(self.op(), x, index, grad)
        expected = vjp(NativeIndexSelectOp(), x, index, grad)
        for a, b in zip(actual, expected, strict=True):
            assert_bits(a, b)

    def test_inner_tiles_exceed_cuda_grid_y_limit(self):
        # Shared selection can copy much larger slices than H3's head widths.
        width = 65535 * 256 + 1
        x = torch.ones(1, 2, 1, device="cuda", dtype=torch.float16).expand(1, 2, width)
        index = torch.tensor([1], device="cuda")
        grad = torch.ones(1, 1, 1, device="cuda", dtype=x.dtype).expand(1, 1, width)
        output, dx = vjp(self.op(), x, index, grad)
        assert_bits(output, grad)
        assert_bits(dx[:, 1:2], grad)
        assert torch.count_nonzero(dx[:, 0]) == 0

    def test_nondefault_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = torch.randn(2, 67, 96, device="cuda")
            index = torch.tensor([66, 0, 66, 5, 0], device="cuda").repeat(33)
            grad = torch.randn(2, index.numel(), 96, device="cuda")
            output, dx = vjp(self.op(), x, index, grad)
        torch.cuda.current_stream().wait_stream(stream)
        expected = vjp(NativeIndexSelectOp(), x, index, grad)
        assert_bits(output, expected[0])
        assert_bits(dx, expected[1])

    def test_device_guards_deterministic_mode_and_runtime(self):
        from rl_engine.ops.autograd.backward_runtime import snapshot_backward_runtime
        from rl_engine.runtime.registry import KernelRegistry

        op = KernelRegistry().get_op("index_select", device="cuda")
        x = torch.randn(1, 7, 32, device="cuda")
        index = torch.tensor([6, 0, 6], device="cuda")
        grad = torch.randn(1, 3, 32, device="cuda")
        expected = vjp(op, x, index, grad)
        old = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            actual = vjp(op, x, index, grad)
        finally:
            torch.use_deterministic_algorithms(old)
        for a, b in zip(actual, expected, strict=True):
            assert_bits(a, b)
        event = snapshot_backward_runtime()["index_select"]
        assert event["family"] == "triton" and "stable_segment" in event["impl"]
        with pytest.raises(ValueError, match="same device"):
            op(x, index.cpu(), dim=1)
        with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
            op(x.cpu(), index.cpu(), dim=1)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two NVIDIA GPUs")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("dim", [0, 1])
    def test_noncurrent_device_and_restoration(self, dtype, dim):
        op = self.op()
        with torch.cuda.device(0):
            x = torch.randn(7, 7, 9, device="cuda:1", dtype=dtype)
            index = torch.tensor([6, 0, 6], device="cuda:1")
            grad = torch.randn_like(x.index_select(dim, index))
            output, dx = vjp(op, x, index, grad, dim=dim)
            assert torch.cuda.current_device() == 0
            assert output.device == dx.device == x.device
            expected = vjp(NativeIndexSelectOp(), x, index, grad, dim=dim)
            assert_bits(output, expected[0])
            assert_bits(dx, expected[1])

    @pytest.mark.parametrize(
        "shape,bad_index",
        [
            ((1, 7, 32), -1),
            ((1, 7, 32), 7),
            ((1, 7, 32), 2**63 - 1),
            ((0, 7, 32), 7),
            ((1, 7, 0), -1),
        ],
    )
    def test_invalid_cuda_index_fails_in_isolated_process(self, shape, bad_index):
        code = f"""
import torch
from rl_engine.backends.shared.triton.embedding.index_select import TritonIndexSelectOp
x = torch.zeros({shape!r}, device='cuda')
i = torch.tensor([{bad_index}], device='cuda')
TritonIndexSelectOp()(x, i, dim=1)
torch.cuda.synchronize()
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=90,
            env={**os.environ, "TRITON_DEBUG": "0"},
        )
        assert result.returncode != 0
        assert "device-side assert" in result.stderr or "index values must be" in result.stderr
