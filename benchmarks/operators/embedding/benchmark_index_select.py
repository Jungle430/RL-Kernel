# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Public index_select/autograd benchmarks with all-element accuracy gates.

Run from the checkout with PYTHONPATH=.; --providers cuda requires the extension.
JSON includes every sample, source hashes, environment and failing comparisons.
Native PyTorch and its deterministic-mode variant are performance baselines;
the CPU FP64 reference is an accuracy oracle and is never timed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.ops.autograd.backward_runtime import snapshot_backward_runtime
from rl_engine.reference.embedding.index_select import NativeIndexSelectOp

ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATHS = (
    "rl_engine/reference/embedding/index_select.py",
    "rl_engine/backends/shared/triton/embedding/index_select.py",
    "rl_engine/backends/cuda/embedding/index_select.py",
    "csrc/cuda/embedding/index_select_h100.cu",
    "benchmarks/operators/embedding/benchmark_index_select.py",
    "rl_engine/contracts/profiles/precision/ws1.json",
)


def make_inputs(shape, dim, duplicate_fraction, dtype, device, seed):
    """Select half the requested axis, with a declared fraction of repeated indices.

    These are synthetic operator workloads. A shape matching H3 or BAGEL does
    not constitute full-model integration or a captured modality distribution.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    axis_size = shape[dim]
    count = max(1, axis_size // 2)
    unique = max(1, round(count * (1 - duplicate_fraction)))
    destinations = torch.randperm(axis_size, generator=generator)[:unique]
    index = destinations[torch.arange(count) % unique]
    index = index[torch.randperm(count, generator=generator)]
    output_shape = list(shape)
    output_shape[dim] = count
    x = torch.randn(shape, generator=generator).to(dtype=dtype, device=device)
    grad = torch.randn(output_shape, generator=generator).to(dtype=dtype, device=device)
    return x, index.to(device), grad


def run_vjp(op, x, index, grad):
    leaf = x.detach().requires_grad_(True)
    output = op(leaf, index)
    return output.detach(), torch.autograd.grad(output, leaf, grad)[0]


def bit_mismatches(actual, expected):
    integer = torch.int32 if actual.element_size() == 4 else torch.int16
    return int((actual.contiguous().view(integer) != expected.contiguous().view(integer)).sum())


def check_result(op, x, index, grad, gold, tolerance, *, dim, ordered_dx=None):
    output, dx = run_vjp(op, x, index, grad)
    repeated_y, repeated_dx = run_vjp(op, x, index, grad)
    error = (dx.float() - gold).abs()
    bad = ~torch.isfinite(error) | (error > tolerance.atol + tolerance.rtol * gold.abs())
    checks = {
        "forward_bit_mismatches": bit_mismatches(output, x.index_select(dim, index)),
        "gradient_accuracy_failures": int(bad.sum()),
        "gradient_max_abs_error": float(error.max()),
        "repeat_forward_bit_mismatches": bit_mismatches(output, repeated_y),
        "repeat_gradient_bit_mismatches": bit_mismatches(dx, repeated_dx),
        "gradient_atol": tolerance.atol,
        "gradient_rtol": tolerance.rtol,
    }
    if ordered_dx is not None:
        checks["ordered_gradient_bit_mismatches"] = bit_mismatches(dx, ordered_dx)
    return checks


def checks_failed(checks):
    """Keep numerical accuracy, copied bits, repeatability and ordered VJP separate."""
    return any(value for name, value in checks.items() if name.endswith(("mismatches", "failures")))


def measure(op, x, index, grad, warmup, samples):
    """Include allocations, validation, wrapping and sorting in the owning phase.

    Forward runs with autograd enabled. Backward reuses a retained forward graph;
    F+B constructs a new graph each time. Input creation, gold, warmups and JIT
    compilation are excluded. No .grad buffers accumulate between repetitions.
    """
    leaf = x.detach().requires_grad_(True)
    output = op(leaf, index)

    def forward():
        return op(leaf, index)

    def backward():
        return torch.autograd.grad(output, leaf, grad, retain_graph=True)[0]

    def combined():
        return torch.autograd.grad(op(leaf, index), leaf, grad)[0]

    timings = {}
    for phase, fn in (("forward", forward), ("backward", backward), ("forward_backward", combined)):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        event_ms, wall_ms = [], []
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(samples):
            start = time.perf_counter()
            begin.record()
            fn()
            end.record()
            end.synchronize()
            wall_ms.append((time.perf_counter() - start) * 1000)
            event_ms.append(begin.elapsed_time(end))
        timings[phase] = {
            "event_ms": event_ms,
            "wall_ms": wall_ms,
            "median_event_ms": statistics.median(event_ms),
            "median_wall_ms": statistics.median(wall_ms),
        }
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    combined()
    torch.cuda.synchronize()
    timings["incremental_peak_bytes"] = torch.cuda.max_memory_allocated() - allocated
    return timings


def measure_graph(op, x, index, grad, expected_y, expected_dx, *, samples=5, calls=4, replays=10):
    """Separate device replay time from the public eager API's host overhead.

    Capture the same public forward/autograd calls as measure(). Allocations,
    validation and grouping kernels belong to the capture; Python dispatch and
    host allocation do not run during replay. Compare captured/replayed results
    with independent supplied references. Native PyTorch failures stay visible.
    """
    leaf = x.detach().requires_grad_(True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    measurements = {}
    with torch.cuda.stream(stream):
        output = op(leaf, index)
        callbacks = {
            "forward": lambda: op(leaf, index),
            "backward": lambda: torch.autograd.grad(output, leaf, grad, retain_graph=True)[0],
            "forward_backward": lambda: torch.autograd.grad(op(leaf, index), leaf, grad)[0],
        }
        for phase, callback in callbacks.items():
            for _ in range(3):
                callback()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(calls):
                    captured = callback()
            for _ in range(3):
                graph.replay()
            stream.synchronize()
            expected = expected_y if phase == "forward" else expected_dx
            before = bit_mismatches(captured, expected)
            times = []
            for _ in range(samples):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                for _ in range(replays):
                    graph.replay()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end) * 1000 / (calls * replays))
            measurements[phase] = {
                "samples_us": times,
                "median_us": statistics.median(times),
                "min_us": min(times),
                "max_us": max(times),
                "ordered_bit_mismatches_before": before,
                "ordered_bit_mismatches_after": bit_mismatches(captured, expected),
            }
    torch.cuda.current_stream().wait_stream(stream)
    return measurements


def command_output(*args):
    try:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def environment():
    import triton

    props = torch.cuda.get_device_properties(0)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": props.name,
        "sm": f"{props.major}{props.minor}",
        "total_memory_bytes": props.total_memory,
        "driver": command_output(
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ),
        "commit": command_output("git", "rev-parse", "HEAD"),
        "git_status": command_output("git", "status", "--porcelain"),
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCE_PATHS
        },
    }


def parse_shape(value):
    """Parse explicit dimensions such as 2x129x96 or 129x28x128."""
    try:
        shape = tuple(int(size) for size in value.split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must contain integers separated by x") from exc
    if not shape or min(shape) < 1:
        raise argparse.ArgumentTypeError("benchmark dimensions must be positive")
    return shape


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes", type=parse_shape, nargs="+", default=[(1, 129, 32), (1, 4097, 96)]
    )
    parser.add_argument(
        "--dim", type=int, required=True, help="Shared selection axis; negative allowed."
    )
    parser.add_argument("--duplicates", type=float, nargs="+", default=[0.0, 0.25])
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["torch", "torch_deterministic", "triton", "cuda", "auto"],
        default=["torch", "torch_deterministic", "triton"],
    )
    parser.add_argument("--mode", choices=["eager", "graph", "both"], default="eager")
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--output", type=Path, default=Path("reports/index-select/results.json"))
    parser.add_argument("--dry-run", action="store_true", help="Print the case plan without a GPU.")
    args = parser.parse_args(argv)
    if args.samples < 1 or args.warmup < 1:
        parser.error("samples and warmup must be positive")
    if any(not -len(shape) <= args.dim < len(shape) for shape in args.shapes):
        parser.error("dim must be a valid axis for every shape")
    if any(not 0 <= fraction < 1 for fraction in args.duplicates):
        parser.error("duplicate fractions must be in [0, 1)")
    return args


def main():
    args = parse_args()
    case_plan = [
        {"shape": list(shape), "dim": args.dim, "duplicate_fraction": fraction}
        for shape in args.shapes
        for fraction in args.duplicates
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "cases": case_plan,
                    "measurement_count": len(case_plan)
                    * len(args.providers)
                    * 3
                    * (2 if args.mode == "both" else 1),
                },
                indent=2,
            )
        )
        return
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("benchmark requires NVIDIA CUDA")
    from rl_engine.backends.shared.triton.embedding.index_select import TritonIndexSelectOp

    dtype = getattr(torch, args.dtype)
    tolerance = resolve_tolerance(
        load_contract(), judgment="gradient_accuracy", op_class="reduction", dtype=dtype
    )
    implementations = [
        ("torch", lambda x, index: x.index_select(args.dim, index), False),
        ("torch_deterministic", lambda x, index: x.index_select(args.dim, index), True),
        ("triton", partial(TritonIndexSelectOp(), dim=args.dim), False),
    ]
    if "cuda" in args.providers:
        from rl_engine.backends.cuda.embedding.index_select import H100IndexSelectOp

        implementations.append(("cuda", partial(H100IndexSelectOp(), dim=args.dim), False))
    if "auto" in args.providers:
        from rl_engine.runtime.registry import KernelRegistry

        selected = KernelRegistry().get_op("index_select", device="cuda")
        implementations.append(("auto", partial(selected, dim=args.dim), False))
    implementations = [entry for entry in implementations if entry[0] in args.providers]
    report = {
        "environment": environment(),
        "arguments": {**vars(args), "output": str(args.output)},
        "method": {
            "mode": args.mode + ", autograd-enabled forward",
            "included": "validation, allocations, wrappers, backward index grouping and zero fill",
            "excluded": "inputs, reference, JIT compilation, warmups",
            "backward": "prebuilt retained graph; index grouping repeated on each backward",
            "wall_ms": "host dispatch plus execution and end synchronization",
            "event_ms": "CUDA events around the complete call, including device idle dispatch gaps",
            "reference": "CPU FP64 accuracy oracle and ordered FP32 VJP reference, neither timed",
            "comparison": "all elements; shared WS1 gradient_accuracy/reduction thresholds",
            "selection": "synthetic half-axis selection; explicit shapes and dim are recorded",
            "graph": "four captured calls/replay; ten replays/sample; five samples; device only",
        },
        "complete": False,
        "case_plan": case_plan,
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        for case in case_plan:
            x, index, grad = make_inputs(
                case["shape"], args.dim, case["duplicate_fraction"], dtype, "cuda", args.seed
            )
            reference = NativeIndexSelectOp()
            gold = run_vjp(
                partial(reference.forward_fp32, dim=args.dim), x.float(), index, grad.float()
            )[1]
            ordered_y, ordered_dx = run_vjp(partial(reference, dim=args.dim), x, index, grad)
            for name, op, deterministic in implementations:
                torch.use_deterministic_algorithms(deterministic)
                checks = check_result(
                    op,
                    x,
                    index,
                    grad,
                    gold,
                    tolerance,
                    dim=args.dim,
                    ordered_dx=ordered_dx,
                )
                result = {
                    **case,
                    "implementation": name,
                    "deterministic_algorithms": deterministic,
                    "selected_count": index.numel(),
                    "duplicate_fraction_actual": 1 - index.unique().numel() / index.numel(),
                    "checks": checks,
                }
                report["results"].append(result)
                if name in ("triton", "cuda", "auto") and checks_failed(checks):
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    raise RuntimeError(f"candidate checks failed before timing: {args.output}")
                if args.mode in ("eager", "both"):
                    result["timings"] = measure(op, x, index, grad, args.warmup, args.samples)
                if args.mode in ("graph", "both"):
                    result["graph_timings"] = measure_graph(
                        op, x, index, grad, ordered_y, ordered_dx
                    )
                    if name in ("triton", "cuda", "auto") and any(
                        phase["ordered_bit_mismatches_before"]
                        or phase["ordered_bit_mismatches_after"]
                        for phase in result["graph_timings"].values()
                    ):
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
                        raise RuntimeError(f"captured candidate checks failed: {args.output}")
                report["backward_runtime"] = snapshot_backward_runtime()
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({k: v for k, v in result.items() if k != "timings"}), flush=True)
    finally:
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
