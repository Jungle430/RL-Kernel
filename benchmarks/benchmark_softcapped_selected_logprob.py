# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compare eager PyTorch, split Triton, and two fused forward implementations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the softcap benchmark's profiler timing, peak-memory measurement and
# environment fingerprint. Only the workloads and report differ here.
from benchmarks.benchmark_final_logit_softcap import (  # noqa: E402
    DTYPES,
    MODES,
    _environment,
    _measure,
    _positive_int,
)
from benchmarks.profiler import PerformanceProfiler  # noqa: E402
from rl_engine.kernels.gtest.tolerance import load_contract, resolve_tolerance  # noqa: E402
from rl_engine.kernels.ops.pytorch.loss import NativeSoftcappedSelectedLogprobOp  # noqa: E402

DEFAULT_SHAPES = ("1x1025", "1x262144", "4x262144", "16x262144", "64x262144")


def _shape(value: str) -> tuple[int, int]:
    try:
        dims = tuple(int(dim) for dim in value.split("x"))
        if len(dims) != 2 or any(dim <= 0 for dim in dims):
            raise ValueError
        return dims
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected positive MxV, e.g. 16x262144") from exc


class _SplitTritonOp:
    """Materialize FP32 softcap output before the existing Triton logprob op."""

    def __init__(self):
        from rl_engine.kernels.ops.triton.activation import TritonFinalLogitSoftcapOp
        from rl_engine.kernels.ops.triton.loss.batch_invariant_logp import (
            TritonBatchInvariantLogpOp,
        )

        self.softcap = TritonFinalLogitSoftcapOp()
        self.logprob = TritonBatchInvariantLogpOp()

    def __call__(self, logits, token_ids):
        return self.logprob(self.softcap(logits), token_ids)


def _make_workload(op, logits, token_ids, upstream, mode):
    """Prepare inputs/graphs outside timing; never accumulate into leaf.grad."""
    if mode == "forward":

        @torch.no_grad()
        def forward():
            op(logits, token_ids)

        return forward

    leaf = logits.detach().requires_grad_(True)
    if mode == "backward":
        output = op(leaf, token_ids)

        def backward():
            torch.autograd.grad(output, leaf, grad_outputs=upstream, retain_graph=True)

        return backward
    if mode == "forward_backward":

        def forward_backward():
            torch.autograd.grad(op(leaf, token_ids), leaf, grad_outputs=upstream)

        return forward_backward
    raise ValueError(f"unknown mode: {mode}")


def _check_accuracy(native, candidate, logits, token_ids, upstream):
    def evaluate(op):
        leaf = logits.detach().requires_grad_(True)
        output = op(leaf, token_ids)
        gradient = torch.autograd.grad(output, leaf, grad_outputs=upstream)[0]
        return output.detach(), gradient

    expected, actual = evaluate(native), evaluate(candidate)
    errors = {}
    for label, judgment, shape, dtype, reference, result in zip(
        ("output", "gradient"),
        ("forward_accuracy", "gradient_accuracy"),
        (logits.shape[:1], logits.shape),
        (torch.float32, logits.dtype),
        expected,
        actual,
        strict=True,
    ):
        assert result.shape == shape and result.dtype == dtype
        tolerance = resolve_tolerance(
            load_contract(), judgment=judgment, op_class="logprob", dtype=dtype
        )
        torch.testing.assert_close(result, reference, atol=tolerance.atol, rtol=tolerance.rtol)
        errors[label] = {
            "max_abs_error": (result.float() - reference.float()).abs().max().item(),
            "atol": tolerance.atol,
            "rtol": tolerance.rtol,
        }
    return errors


def _check_forward_variants(row_op, parallel_op, logits, token_ids, upstream):
    """Reject changes to the output, saved row statistics or resulting gradient."""

    def evaluate(op):
        leaf = logits.detach().requires_grad_(True)
        output = op(leaf, token_ids)
        _, _, log_sum_exp = output.grad_fn.saved_tensors
        gradient = torch.autograd.grad(output, leaf, grad_outputs=upstream)[0]
        return output.detach(), log_sum_exp, gradient

    expected = evaluate(row_op)
    actual = evaluate(parallel_op)
    checks = {}
    for name, reference, result in zip(
        ("output", "log_sum_exp", "gradient"), expected, actual, strict=True
    ):
        # Compare stored bits, including signed zeros, rather than relaxing the
        # tolerance for the experimental forward's extra global-memory round trip.
        if not torch.equal(
            reference.contiguous().view(torch.uint8), result.contiguous().view(torch.uint8)
        ):
            raise AssertionError(f"Row and parallel forward variants differ in {name}")
        checks[name] = True
    return checks


def _markdown(report):
    env = report["environment"]
    lines = [
        "# Softcapped selected logprob benchmark",
        "",
        f"GPU: {env['gpu']}; {env['backend']} runtime: {env['runtime']}; "
        f"PyTorch: {env['torch']}; Triton: {env['triton']}.",
        f"Commit: `{env['git_commit']}`; tracked changes: "
        f"`{env['git_tracked_changes'] or 'none'}`.",
        f"Warmup: {env['warmup']}; repetitions: {env['repeat']}; "
        f"vocab tile: {env['block_size']}; warps: {env['num_warps']}.",
        "",
        "Native = eager PyTorch. Split = the existing Triton final_logit_softcap followed "
        "by TritonBatchInvariantLogpOp. Fused row = the original per-row loop. "
        "Fused parallel = per-tile partial sums followed by an ordered merge. "
        "Both fused paths share the same tiled backward. No torch.compile or graph capture. "
        "All Triton paths pass output and gradient checks against native before timing; "
        "the fused variants also pass bitwise output, saved log_sum_exp and gradient checks. "
        "The JSON contains errors, tolerances and exact-match results.",
        "Latency is median ± sample standard deviation in ms (N/A for one repetition), "
        "using the profiler's "
        "CUDA/HIP event timer. Public-wrapper allocation and autograd dispatch are included. "
        "The parallel forward allocates scratch and launches both kernels inside the timed "
        "call. Input generation, correctness checks and JIT compilation are outside timing.",
        "Forward uses no_grad. Backward reuses a retained graph; forward+backward builds "
        "a fresh graph per call. Extra peak memory excludes inputs and retained graphs.",
        "Native/row and split/row compare the baselines with fused row. Row/parallel > 1 "
        "means the parallel version is faster; < 1 means it is slower. "
        "Small cases may include substantial host dispatch gaps.",
        "",
        "| Dtype | Shape | Mode | Native ms | Split ms | Fused row ms | Fused parallel ms | "
        "Native/row | Split/row | Row/parallel |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        timings = []
        for name in ("native", "split_triton", "triton", "triton_parallel"):
            measurement = row[name]
            std_ms = measurement["std_ms"]
            std_text = f"{std_ms:.6f}" if std_ms is not None else "N/A"
            timings.append(f"{measurement['median_ms']:.6f} ± {std_text}")
        lines.append(
            f"| {row['dtype']} | {'x'.join(map(str, row['shape']))} | {row['mode']} | "
            + " | ".join(timings)
            + f" | {row['speedup']:.2f}x | {row['split_speedup']:.2f}x | "
            f"{row['parallel_speedup']:.2f}x |"
        )
    lines += [
        "",
        "| Dtype | Shape | Mode | Native extra MiB | Split extra MiB | "
        "Fused row extra MiB | Fused parallel extra MiB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        memory = " | ".join(
            f"{row[name]['peak_extra_mib']:.3f}"
            for name in ("native", "split_triton", "triton", "triton_parallel")
        )
        lines.append(
            f"| {row['dtype']} | {'x'.join(map(str, row['shape']))} | {row['mode']} | {memory} |"
        )
    return "\n".join(lines) + "\n"


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--shapes", nargs="+", type=_shape, default=list(map(_shape, DEFAULT_SHAPES))
    )
    parser.add_argument("--warmup", type=_positive_int, default=10)
    parser.add_argument("--repeat", type=_positive_int, default=50)
    parser.add_argument("--seed", type=int, default=415)
    parser.add_argument("--output-dir", type=Path, default=Path("../softcapped-logprob-results"))
    return parser


def run_benchmark(args):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "This benchmark requires an NVIDIA CUDA or AMD ROCm GPU; no CPU fallback is timed"
        )
    import triton

    from rl_engine.kernels.ops.triton.loss.softcapped_selected_logprob import (
        _BLOCK_V,
        TritonSoftcappedSelectedLogprobOp,
    )

    with torch.cuda.device(device):
        profiler = PerformanceProfiler(device=device, warmup=args.warmup, repeat=args.repeat)
        ops = {
            "native": NativeSoftcappedSelectedLogprobOp(),
            "split_triton": _SplitTritonOp(),
            "triton": TritonSoftcappedSelectedLogprobOp(),
            "triton_parallel": TritonSoftcappedSelectedLogprobOp(forward_impl="parallel"),
        }
        env = _environment(device, args, triton.__version__, _BLOCK_V, profiler.gpu_info)
        env.update(
            {
                "operator": "softcapped_selected_logprob",
                "softcap": 30.0,
                "num_warps": 4,
                "baseline": "eager NativeSoftcappedSelectedLogprobOp; no torch.compile",
                "split_baseline": "TritonFinalLogitSoftcapOp + TritonBatchInvariantLogpOp",
                "fused_schedule": (
                    "forward: one program per row, ascending 1024-element FP32 tile sums; "
                    "backward: one program per row/vocabulary tile using saved log_sum_exp; "
                    "fp_fusion=False"
                ),
                "parallel_fused_schedule": (
                    "forward: one program per 1024-element vocabulary tile, FP32 scratch, "
                    "then one program per row accumulating tile sums in ascending order; "
                    "same backward as fused row; fp_fusion=False"
                ),
                "parallel_scratch_bytes": "4 * M * ceil(V / 1024), forward only",
                "modes": args.modes,
            }
        )
        # Include source hashes even if the measured operator files are untracked.
        sources = [
            "benchmarks/benchmark_softcapped_selected_logprob.py",
            "benchmarks/benchmark_final_logit_softcap.py",
            "rl_engine/kernels/ops/pytorch/loss/softcapped_selected_logprob.py",
            "rl_engine/kernels/ops/triton/loss/softcapped_selected_logprob.py",
            "rl_engine/kernels/ops/triton/activation/final_logit_softcap.py",
            "rl_engine/kernels/ops/triton/loss/batch_invariant_logp.py",
        ]
        env["source_sha256"] = {
            path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest() for path in sources
        }
        report = {"environment": env, "results": []}
        for dtype_name in args.dtypes:
            for shape in args.shapes:
                generator = torch.Generator(device=device).manual_seed(args.seed)
                logits = (torch.randn(shape, device=device, generator=generator) * 30).to(
                    DTYPES[dtype_name]
                )
                ids = torch.randint(shape[1], (shape[0],), device=device, generator=generator)
                upstream = torch.randn(shape[0], device=device, generator=generator)
                accuracy = {
                    name: _check_accuracy(ops["native"], ops[name], logits, ids, upstream)
                    for name in ("split_triton", "triton", "triton_parallel")
                }
                exact_match = _check_forward_variants(
                    ops["triton"], ops["triton_parallel"], logits, ids, upstream
                )
                for mode in args.modes:
                    measurements = {}
                    for name, op in ops.items():
                        fn = _make_workload(op, logits, ids, upstream, mode)
                        measurements[name] = _measure(profiler, fn)
                        del fn
                    fused_ms = measurements["triton"]["median_ms"]
                    row = {
                        "dtype": dtype_name,
                        "shape": list(shape),
                        "mode": mode,
                        "accuracy": accuracy,
                        "forward_variants_bitwise_equal": exact_match,
                        **measurements,
                        "speedup": measurements["native"]["median_ms"] / fused_ms,
                        "split_speedup": measurements["split_triton"]["median_ms"] / fused_ms,
                        "parallel_speedup": fused_ms / measurements["triton_parallel"]["median_ms"],
                    }
                    report["results"].append(row)
                    print(
                        f"{dtype_name} {shape} {mode}: native/row={row['speedup']:.2f}x, "
                        f"split/row={row['split_speedup']:.2f}x, "
                        f"row/parallel={row['parallel_speedup']:.2f}x",
                        flush=True,
                    )
                del logits, ids, upstream
    return report


def main():
    args = build_arg_parser().parse_args()
    report = run_benchmark(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    markdown = _markdown(report)
    (args.output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Reports saved to {args.output_dir}")


if __name__ == "__main__":
    main()
