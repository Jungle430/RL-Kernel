# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU checks for benchmark timing boundaries and its correctness gate."""

import pytest
import torch

from benchmarks.benchmark_softcapped_selected_logprob import (
    _check_accuracy,
    _check_forward_variants,
    _make_workload,
    _markdown,
    build_arg_parser,
    run_benchmark,
)
from rl_engine.kernels.ops.pytorch.loss import NativeSoftcappedSelectedLogprobOp


@pytest.mark.parametrize("mode", ("forward", "backward", "forward_backward"))
def test_workload_timing_and_random_upstream(mode):
    logits = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    ids = torch.tensor([2, 0])
    upstream = torch.tensor([0.5, -2.0])
    native = NativeSoftcappedSelectedLogprobOp()
    gold_leaf = logits.clone().requires_grad_(True)
    expected = torch.autograd.grad(native(gold_leaf, ids), gold_leaf, grad_outputs=upstream)[0]
    calls, gradients = [], []

    def observed(leaf, token_ids):
        calls.append(torch.is_grad_enabled())
        if torch.is_grad_enabled() and len(calls) == 1:
            leaf.register_hook(lambda grad: gradients.append(grad.clone()))
        return native(leaf, token_ids)

    fn = _make_workload(observed, logits, ids, upstream, mode)
    assert len(calls) == (1 if mode == "backward" else 0)
    assert fn() is None and fn() is None
    assert len(calls) == (1 if mode == "backward" else 2)
    if mode == "forward":
        assert calls == [False, False] and not gradients
    else:
        assert len(gradients) == 2
        for grad in gradients:
            torch.testing.assert_close(grad, expected)
    assert logits.grad is None and not logits.requires_grad


def test_benchmark_rejects_cpu():
    args = build_arg_parser().parse_args(["--device", "cpu"])
    with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA or AMD ROCm GPU"):
        run_benchmark(args)


def test_accuracy_gate_rejects_incorrect_gradient():
    class WrongGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, token_ids):
            ctx.shape = logits.shape
            return NativeSoftcappedSelectedLogprobOp()(logits, token_ids)

        @staticmethod
        def backward(ctx, grad_output):
            return torch.ones(ctx.shape, device=grad_output.device), None

    with pytest.raises(AssertionError):
        _check_accuracy(
            NativeSoftcappedSelectedLogprobOp(),
            WrongGradient.apply,
            torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]),
            torch.tensor([2, 0]),
            torch.tensor([0.5, -2.0]),
        )


@pytest.mark.parametrize("std_ms", (0.1, None))
def test_report_discloses_split_baseline_and_regression(std_ms):
    env = dict(
        gpu="test GPU",
        backend="cuda",
        runtime="test",
        torch="test",
        triton="test",
        git_commit="test",
        git_tracked_changes="",
        warmup=1,
        repeat=2 if std_ms is not None else 1,
        block_size=1024,
        num_warps=4,
    )
    measurement = dict(median_ms=2.0, std_ms=std_ms, peak_extra_mib=0.5)
    parallel_std = 0.2 if std_ms is not None else None
    report = {
        "environment": env,
        "results": [
            {
                "dtype": "fp32",
                "shape": [1, 1025],
                "mode": "forward",
                "speedup": 0.5,
                "split_speedup": 0.75,
                "parallel_speedup": 0.8,
                "native": measurement,
                "split_triton": measurement,
                "triton": measurement,
                "triton_parallel": dict(median_ms=2.5, std_ms=parallel_std, peak_extra_mib=0.75),
            }
        ],
    }
    markdown = _markdown(report)
    assert "TritonBatchInvariantLogpOp" in markdown
    assert "0.50x" in markdown and "0.75x" in markdown
    assert "0.80x" in markdown
    expected_std = "0.200000" if parallel_std is not None else "N/A"
    assert f"2.500000 ± {expected_std}" in markdown
    assert "Row/parallel" in markdown and "ordered merge" in markdown
    assert "allocates scratch and launches both kernels inside" in markdown
    assert "standard deviation" in markdown and "extra MiB" in markdown


def test_benchmark_can_time_forward_only():
    args = build_arg_parser().parse_args(["--modes", "forward", "--shapes", "1x262144", "4x262144"])
    assert args.modes == ["forward"]
    assert args.shapes == [(1, 262144), (4, 262144)]


def test_exact_gate_checks_saved_statistics_even_when_outputs_and_gradients_match():
    class WithSavedStatistics(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, token_ids, offset):
            ctx.save_for_backward(logits, token_ids, logits.sum(dim=1) + offset)
            return logits.sum(dim=1)

        @staticmethod
        def backward(ctx, upstream):
            logits, _, _ = ctx.saved_tensors
            return upstream[:, None].expand_as(logits), None, None

    logits = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    ids = torch.tensor([2, 0])
    upstream = torch.tensor([0.5, -2.0])

    def row_op(x, token_ids):
        return WithSavedStatistics.apply(x, token_ids, 0.0)

    def wrong_statistics(x, token_ids):
        return WithSavedStatistics.apply(x, token_ids, 1.0)

    checks = _check_forward_variants(row_op, row_op, logits, ids, upstream)
    assert checks == {"output": True, "log_sum_exp": True, "gradient": True}
    with pytest.raises(AssertionError, match="differ in log_sum_exp"):
        _check_forward_variants(row_op, wrong_statistics, logits, ids, upstream)


@pytest.mark.parametrize("shape", ("1", "1x2x3", "0x1024", "1x-1"))
def test_benchmark_rejects_invalid_geometry(shape):
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--shapes", shape])
