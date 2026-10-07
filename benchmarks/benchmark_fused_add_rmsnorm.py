# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compare eager PyTorch and both fused-add/RMSNorm weight-gradient strategies.

Time public forward, backward, forward+backward, and the weight reduction alone.
Inputs/residuals share the selected dtype; weights and both upstreams are FP32.
Reports describe measured alternatives only; they do not select a new default.

Example:
    python benchmarks/benchmark_fused_add_rmsnorm.py --rows 1 32 33 128 1024 8192
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _measure(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repeat)
    ]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) for start, end in events]
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    extra_peak_mib = (torch.cuda.max_memory_allocated() - baseline) / (1024 * 1024)
    del result
    return {
        "median_ms": statistics.median(samples),
        "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "samples_ms": samples,
        "extra_peak_mib": extra_peak_mib,
    }


def _check(actual, expected, *, atol=None):
    tolerance = {torch.float16: 3e-3, torch.bfloat16: 2e-2, torch.float32: 2e-5}[actual.dtype]
    torch.testing.assert_close(
        actual, expected.to(actual.dtype), rtol=tolerance, atol=tolerance if atol is None else atol
    )
    return (actual.double() - expected.double()).abs().max().item()


def _run_case(args, module, triton, n_rows, n_cols, dtype_name):
    from rl_engine.kernels.ops.pytorch.norm.fused_add_rmsnorm import NativeFusedAddRMSNormOp

    # Reset the seed per case so comparing a subset with a full sweep reuses its data.
    torch.manual_seed(args.seed)
    shape = (n_rows, n_cols)
    dtype = _DTYPES[dtype_name]
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    residual = torch.randn_like(x, requires_grad=True)
    weight = torch.randn(n_cols, device="cuda", dtype=torch.float32, requires_grad=True)
    inputs = (x, residual, weight)
    upstream = tuple(torch.randn(shape, device="cuda", dtype=torch.float32) for _ in range(2))
    native = NativeFusedAddRMSNormOp()
    native_outputs = native(*inputs)
    reference = torch.autograd.grad(native_outputs, inputs, upstream)
    # Long sequential FP32 sums accumulate error. Record this explicit, row-scaled
    # absolute tolerance as well as the actual errors, rather than asserting bitwise dw.
    weight_atol = 2e-5 * math.sqrt(n_rows)
    providers = {"native": native}
    providers.update(
        {
            strategy.value: module.TritonFusedAddRMSNormOp(weight_grad_strategy=strategy)
            for strategy in module.RMSNormWeightGradStrategy
        }
    )
    results = []
    per_row = None
    metadata = {
        "shape": list(shape),
        "input_dtype": dtype_name,
        "weight_dtype": "fp32",
        "weight_grad_atol": weight_atol,
    }
    for name, op in providers.items():
        outputs = op(*inputs)
        gradients = torch.autograd.grad(outputs, inputs, upstream, retain_graph=True)
        errors = {
            "y": _check(outputs[0], native_outputs[0]),
            "updated_residual": _check(outputs[1], native_outputs[1]),
            "grad_x": _check(gradients[0], reference[0]),
            "grad_residual": _check(gradients[1], reference[1]),
            "grad_weight": _check(gradients[2], reference[2], atol=weight_atol),
        }
        if name == "sequential":
            updated, inverse_rms, _ = outputs[0].grad_fn.saved_tensors
            with torch.no_grad():
                per_row = upstream[0] * (updated * inverse_rms[:, None])

        def backward(outputs=outputs):
            return torch.autograd.grad(outputs, inputs, upstream, retain_graph=True)

        def forward(op=op):
            with torch.no_grad():
                return op(*inputs)

        def forward_backward(op=op):
            return torch.autograd.grad(op(*inputs), inputs, upstream)

        for mode, fn in (
            ("forward", forward),
            ("backward", backward),
            ("forward_backward", forward_backward),
        ):
            results.append(
                {
                    **metadata,
                    "provider": name,
                    "mode": mode,
                    "max_abs_errors_vs_native": errors,
                    **_measure(fn, args.warmup, args.repeat),
                }
            )

    # Both kernels read the same precomputed FP32 contributions into preallocated
    # outputs. No operator wrapper, allocation or autograd is included in this scope.
    expected_weight = per_row.double().sum(dim=0)
    reduced = torch.empty_like(weight)
    grid = (triton.cdiv(n_cols, module._WEIGHT_BLOCK_SIZE),)
    reducers = {"native": lambda: torch.sum(per_row, dim=0, out=reduced)}
    for strategy, kernel in module._WEIGHT_GRAD_KERNELS.items():

        def reduce_weight(kernel=kernel):
            kernel[grid](
                per_row,
                reduced,
                n_rows,
                n_cols,
                BLOCK_SIZE=module._WEIGHT_BLOCK_SIZE,
                num_warps=module._NUM_WARPS,
                enable_fp_fusion=False,
            )

        reducers[strategy.value] = reduce_weight
    for name, fn in reducers.items():
        fn()
        error = _check(reduced, expected_weight, atol=weight_atol)
        results.append(
            {
                **metadata,
                "provider": name,
                "mode": "weight_reduction",
                "max_abs_error_vs_fp64": error,
                **_measure(fn, args.warmup, args.repeat),
            }
        )
    return results


def _report(payload):
    env = payload["environment"]
    lines = [
        "# Fused add RMSNorm strategy benchmark",
        "",
        f"GPU: {env['gpu']}; PyTorch: {env['torch']}; Triton: {env['triton']}.",
        f"Complete: {payload['complete']}; measurements: "
        f"{len(payload['results'])}/{payload['case_plan']['measurement_count']}.",
        f"Warmup: {payload['config']['warmup']}; repetitions: {payload['config']['repeat']}; "
        f"seed: {payload['config']['seed']}.",
        "",
        "Eager PyTorch vs sequential and experimental tiled Triton reductions. "
        "Weights/upstreams and intermediate arithmetic are FP32. Tiled uses 32 rows x 128 columns. "
        "No torch.compile, CUDA Graph capture or explicit cache eviction. "
        "Median accelerator-event timings exclude compilation and correctness checks; "
        "public timings include wrapper allocation/dispatch and may include host gaps. "
        "Forward uses no_grad. "
        "Backward reuses a graph; forward+backward creates a fresh graph per call. "
        "Weight reduction alone uses identical contributions and preallocated outputs; "
        "its native baseline is torch.sum, not a full operator call. "
        "Input/output/gradient checks run before timing. Weight-gradient atol is "
        "2e-5 * sqrt(rows), rtol is 2e-5; JSON records errors, tolerances and timing samples. "
        "Changed addition order is not a guarantee of bitwise equality between strategies.",
        "Speedup = eager PyTorch / Triton; values below 1 mean Triton was slower. "
        "Extra peak allocations exclude inputs and prebuilt graphs; full samples, standard "
        "deviations, errors and allocation measurements are in results.json.",
        "",
        "| Input | Shape | Scope | Native ms | Sequential ms | Tiled ms | "
        "Native / seq | Native / tiled | Seq / tiled |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    grouped = {}
    for result in payload["results"]:
        key = (result["input_dtype"], tuple(result["shape"]), result["mode"])
        grouped.setdefault(key, {})[result["provider"]] = result["median_ms"]
    for (dtype, shape, mode), times in grouped.items():
        native, sequential, tiled = (times[name] for name in ("native", "sequential", "tiled"))
        ratio = f"{sequential / tiled:.2f}x" if tiled > 0 else "n/a"
        seq_speedup = f"{native / sequential:.2f}x" if sequential > 0 else "n/a"
        tiled_speedup = f"{native / tiled:.2f}x" if tiled > 0 else "n/a"
        lines.append(
            f"| {dtype} | {shape[0]}x{shape[1]} | {mode} | {native:.6f} | "
            f"{sequential:.6f} | {tiled:.6f} | {seq_speedup} | {tiled_speedup} | {ratio} |"
        )
    return "\n".join(lines) + "\n"


def _save(payload, folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (folder / "report.md").write_text(_report(payload), encoding="utf-8")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", nargs="+", type=_positive_int, default=[1, 32, 33, 128, 1024, 8192]
    )
    parser.add_argument("--cols", nargs="+", type=_positive_int, default=[129, 2688])
    parser.add_argument("--dtypes", nargs="+", choices=_DTYPES, default=list(_DTYPES))
    parser.add_argument("--warmup", type=_positive_int, default=10)
    parser.add_argument("--repeat", type=_positive_int, default=50)
    parser.add_argument("--seed", type=int, default=434)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/fused-add-rmsnorm"))
    parser.add_argument("--dry-run", action="store_true", help="Print the case plan without a GPU")
    return parser.parse_args(argv)


def _case_plan(args):
    count = len(args.dtypes) * len(args.rows) * len(args.cols)
    return {"case_count": count, "measurement_count": count * 3 * 4}


def main(argv=None):
    args = _parse_args(argv)
    plan = _case_plan(args)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA or ROCm GPU and a working Triton runtime are required")

    import triton

    from rl_engine.kernels.ops.triton.norm import fused_add_rmsnorm as module

    payload = {
        "complete": False,
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "hip": torch.version.hip,
        },
        "config": {**vars(args), "output_dir": str(args.output_dir)},
        "case_plan": plan,
        "results": [],
    }
    _save(payload, args.output_dir)
    for dtype in args.dtypes:
        for n_rows in args.rows:
            for n_cols in args.cols:
                print(f"Checking and timing {dtype} [{n_rows}, {n_cols}]...", flush=True)
                payload["results"].extend(_run_case(args, module, triton, n_rows, n_cols, dtype))
                _save(payload, args.output_dir)
    payload["complete"] = True
    _save(payload, args.output_dir)
    print(_report(payload))
    print(f"Reports saved to {args.output_dir}")


if __name__ == "__main__":
    main()
