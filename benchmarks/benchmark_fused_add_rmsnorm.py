# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compare public RMSNorm calls and diagnose weight-reduction configurations.

Public eager timings include Python dispatch, allocation, and autograd. Separate
CUDA Graph measurements amortize host launches and are never mixed with eager
speedups. --sweep-configs sweeps reduction kernels, then checks the best measured
configuration of each strategy through the public operator on the same input.
This is an experiment, not an automatic production dispatch policy.
"""

import argparse
import gc
import itertools
import json
import math
import statistics
import time
from functools import partial
from pathlib import Path

import torch

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
_DEFAULT_CONFIGS = {
    "sequential": (1, 128, 4),
    "tiled": (32, 128, 4),
    "parallel": (256, 128, 4),
}
_PRESETS = {
    "smoke": ([1, 33], [129, 2688]),
    "model": ([1, 32, 33, 128, 1024, 8192], [129, 2688]),
    "tuning": ([1, 16, 31, 32, 33, 128, 512, 1024, 2048, 8192, 32768], [128, 129, 2688, 4096]),
}


def _positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _variants(sweep):
    variants = []
    seen = set()

    def add(name, strategy, rows, cols, warps):
        key = (strategy, rows, cols, warps)
        if key not in seen:
            seen.add(key)
            variants.append(
                dict(
                    name=name, strategy=strategy, block_rows=rows, block_cols=cols, num_warps=warps
                )
            )

    for strategy, config in _DEFAULT_CONFIGS.items():
        add(strategy, strategy, *config)
    if sweep:
        for strategy, rows, cols, warps in itertools.chain(
            (("sequential", 1, c, w) for c, w in itertools.product((32, 64, 128, 256), (4, 8))),
            (("tiled", r, c, w) for r, c, w in itertools.product((16, 32, 64), (64, 128), (4, 8))),
            (
                ("parallel", r, c, w)
                for r, c, w in itertools.product((64, 128, 256, 512), (64, 128), (4, 8))
            ),
        ):
            add(f"{strategy}-r{rows}-c{cols}-w{warps}", strategy, rows, cols, warps)
    return variants


def _round_order(names, round_index):
    # Paired reversed orders balance first/last position, then rotate the pair.
    offset = (round_index // 2) % len(names)
    order = names[offset:] + names[:offset]
    return order if round_index % 2 == 0 else order[::-1]


def _summarize(rounds):
    samples = [sample for r in rounds for sample in r["samples_ms"]]
    medians = [r["median_ms"] for r in rounds]
    return {
        "median_ms": statistics.median(medians),
        "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "round_spread": max(medians) / min(medians),
        "wall_ms_per_call": statistics.median(r["wall_ms_per_call"] for r in rounds),
        "samples_ms": samples,
        "rounds": rounds,
    }


def _measure_block(fn, repeat, divisor):
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repeat)
    ]
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - begin) * 1000 / (repeat * divisor)
    samples = [start.elapsed_time(end) / divisor for start, end in events]
    if not all(math.isfinite(v) and v > 0 for v in samples):
        raise RuntimeError("Nonpositive or nonfinite timer sample; increase --graph-unroll.")
    return dict(samples_ms=samples, median_ms=statistics.median(samples), wall_ms_per_call=wall_ms)


def _capture(fn, unroll):
    # Compile/warm first, capture on a side stream, then replay many GPU calls
    # per host launch. Inputs and captured buffers remain alive through timing.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(unroll):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _measure_group(functions, args, *, timing):
    for fn in functions.values():
        for _ in range(args.warmup):
            fn()
    torch.cuda.synchronize()
    graphs = (
        {name: _capture(fn, args.graph_unroll) for name, fn in functions.items()}
        if timing == "graph"
        else {}
    )
    calls = {name: graph.replay for name, graph in graphs.items()} if graphs else functions
    divisor = args.graph_unroll if graphs else 1
    rounds = {name: [] for name in calls}
    orders = []
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for round_index in range(args.rounds):
            order = _round_order(list(calls), round_index)
            orders.append(order)
            for position, name in enumerate(order):
                for _ in range(args.warmup):
                    calls[name]()
                record = _measure_block(calls[name], args.repeat, divisor)
                rounds[name].append(dict(index=round_index, position=position, **record))
    finally:
        if gc_enabled:
            gc.enable()
    return {name: _summarize(records) for name, records in rounds.items()}, orders


def _extra_peak(fn):
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - baseline) / (1024 * 1024)
    del result
    return peak


def _check(actual, expected, *, atol=None):
    tolerance = {torch.float16: 3e-3, torch.bfloat16: 2e-2, torch.float32: 2e-5}[actual.dtype]
    torch.testing.assert_close(
        actual, expected.to(actual.dtype), rtol=tolerance, atol=tolerance if atol is None else atol
    )
    return (actual.double() - expected.double()).abs().max().item()


def _config(module, spec):
    return module.RMSNormWeightGradConfig(
        block_cols=spec["block_cols"], block_rows=spec["block_rows"], num_warps=spec["num_warps"]
    )


def _prepare_reducers(module, specs, per_row, weight):
    n_rows, n_cols = per_row.shape
    # Independent outputs prevent one captured graph from overwriting another's
    # result. Only the immutable source contributions are shared.
    outputs = {name: torch.empty_like(weight) for name in ["native"] + [s["name"] for s in specs]}
    functions = {"native": partial(torch.sum, per_row, dim=0, out=outputs["native"])}
    metadata = {"native": {"workspace_mib": 0.0}}
    for spec in specs:
        strategy = module.RMSNormWeightGradStrategy(spec["strategy"])
        config = _config(module, spec)
        kwargs = dict(
            grad_weight_per_row=per_row,
            grad_weight=outputs[spec["name"]],
            n_rows=n_rows,
            n_cols=n_cols,
            config=config,
        )
        workspace_mib = 0.0
        col_programs = (n_cols + config.block_cols - 1) // config.block_cols
        partial_count = 0
        if strategy == module.RMSNormWeightGradStrategy.PARALLEL:
            partial_count = (n_rows + config.block_rows - 1) // config.block_rows
            partials = torch.empty(
                (partial_count, n_cols),
                device=per_row.device,
                dtype=torch.float32,
            )
            kwargs["partials"] = partials
            workspace_mib = partials.numel() * partials.element_size() / (1024 * 1024)
        functions[spec["name"]] = partial(module._WEIGHT_GRAD_LAUNCHERS[strategy], **kwargs)
        metadata[spec["name"]] = {
            **spec,
            "workspace_mib": workspace_mib,
            "partial_count": partial_count,
            "partial_program_count": col_programs * partial_count,
            "final_program_count": col_programs,
        }
    return functions, outputs, metadata


def _run_case(args, module, n_rows, n_cols, dtype_name):
    from rl_engine.kernels.ops.pytorch.norm.fused_add_rmsnorm import NativeFusedAddRMSNormOp

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
    weight_atol = 2e-5 * math.sqrt(n_rows)
    metadata = dict(
        shape=list(shape),
        input_dtype=dtype_name,
        weight_dtype="fp32",
        eps=1e-5,
        weight_grad_atol=weight_atol,
    )
    records, order_records = [], []

    def measure(functions, mode, timing, extra):
        print(f"  {mode}/{timing}: {len(functions)} providers, {args.rounds} rounds", flush=True)
        measured, orders = _measure_group(functions, args, timing=timing)
        order_records.append(dict(mode=mode, timing=timing, orders=orders))
        for name, stats in measured.items():
            records.append(
                {
                    **metadata,
                    "provider": name,
                    "mode": mode,
                    "timing": timing,
                    **extra[name],
                    **stats,
                }
            )
        return measured

    # Derive identical FP32 contributions independently of any Triton backward.
    with torch.no_grad():
        updated = x.float() + residual.float()
        inverse_rms = torch.rsqrt(updated.square().mean(dim=-1, keepdim=True) + 1e-5)
        per_row = upstream[0] * (updated * inverse_rms)
        expected_weight = per_row.double().sum(dim=0)
    specs = _variants(args.sweep_configs)
    reducers, reduced, reducer_meta = _prepare_reducers(module, specs, per_row, weight)
    for name, fn in reducers.items():
        fn()
        error = _check(reduced[name], expected_weight, atol=weight_atol)
        first = reduced[name].clone()
        reduced[name].fill_(float("nan"))
        fn()
        repeat_equal = torch.equal(first.view(torch.uint8), reduced[name].view(torch.uint8))
        if not repeat_equal:
            raise AssertionError(f"Non-repeatable weight reduction: {name}")
        reducer_meta[name].update(max_abs_error_vs_fp64=error, repeat_bitwise=True)
    measure(reducers, "weight_reduction", "eager", reducer_meta)
    kernel_times = measure(reducers, "weight_reduction", "graph", reducer_meta)
    # Graph capture/replay must also leave the expected outputs in the buffers.
    for name in reducers:
        _check(reduced[name], expected_weight, atol=weight_atol)

    public_specs = [dict(s) for s in specs[:3]]
    selections = {}
    if args.sweep_configs:
        for strategy in _DEFAULT_CONFIGS:
            candidates = [s for s in specs if s["strategy"] == strategy]
            best = min(candidates, key=lambda s: kernel_times[s["name"]]["median_ms"])
            selections[strategy] = dict(best)
            public_specs.append({**best, "name": f"{strategy}_tuned"})
    # Drop isolated workspaces before measuring public peak allocation.
    del reducers, reduced, per_row, updated, inverse_rms, fn, first
    providers = {"native": native}
    provider_meta = {"native": {}}
    for spec in public_specs:
        providers[spec["name"]] = module.TritonFusedAddRMSNormOp(
            weight_grad_strategy=module.RMSNormWeightGradStrategy(spec["strategy"]),
            weight_grad_config=_config(module, spec),
        )
        provider_meta[spec["name"]] = dict(spec)
    functions = {mode: {} for mode in ("forward", "backward", "forward_backward")}
    for name, op in providers.items():
        outputs = op(*inputs)
        gradients = torch.autograd.grad(outputs, inputs, upstream, retain_graph=True)
        errors = {
            label: _check(actual, expected, atol=weight_atol if label == "grad_weight" else None)
            for label, actual, expected in zip(
                ("y", "updated_residual", "grad_x", "grad_residual", "grad_weight"),
                outputs + gradients,
                native_outputs + reference,
                strict=True,
            )
        }
        for context in (torch.no_grad(), torch.inference_mode()):
            with context:
                inference = op(*inputs)
            for training, inferred in zip(outputs, inference, strict=True):
                if not torch.equal(training.view(torch.uint8), inferred.view(torch.uint8)):
                    raise AssertionError(f"Training/inference mismatch: {name}")
        # Sweep candidates also preserve the row-wise numerical path. Weight sums
        # intentionally cover different row sets and are not compared here.
        if name != "native":
            order = torch.tensor([n_rows - 1, 0], device=x.device)
            subset_inputs = tuple(t[order].detach().requires_grad_(True) for t in inputs[:2])
            subset_inputs += (weight,)
            subset_outputs = op(*subset_inputs)
            subset_gradients = torch.autograd.grad(
                subset_outputs, subset_inputs, tuple(t[order] for t in upstream)
            )
            for full, subset in zip(
                outputs + gradients[:2], subset_outputs + subset_gradients[:2], strict=True
            ):
                if not torch.equal(full[order].view(torch.uint8), subset.view(torch.uint8)):
                    raise AssertionError(f"Row invariance mismatch: {name}")
            provider_meta[name]["row_invariance_bitwise"] = True
        provider_meta[name].update(max_abs_errors_vs_native=errors, train_inference_bitwise=True)

        def forward(op=op):
            with torch.no_grad():
                return op(*inputs)

        def backward(outputs=outputs):
            return torch.autograd.grad(outputs, inputs, upstream, retain_graph=True)

        def forward_backward(op=op):
            return torch.autograd.grad(op(*inputs), inputs, upstream)

        functions["forward"][name] = forward
        functions["backward"][name] = backward
        functions["forward_backward"][name] = forward_backward
    for mode, calls in functions.items():
        extra = {
            name: {**provider_meta[name], "extra_peak_mib": _extra_peak(fn)}
            for name, fn in calls.items()
        }
        measure(calls, mode, "eager", extra)
    # Same mathematical forward for every strategy: a device-only baseline helps
    # distinguish forward kernel cost from Python/allocation/dispatch gaps.
    forward_graph = {name: functions["forward"][name] for name in ("native", "sequential")}
    measure(forward_graph, "forward", "graph", {n: provider_meta[n] for n in forward_graph})
    forward_times = [
        r["median_ms"]
        for r in records
        if r["mode"] == "forward" and r["timing"] == "eager" and r["provider"] != "native"
    ]
    control_ratio = max(forward_times) / min(forward_times)
    diagnostic = dict(
        **metadata,
        same_forward_spread=control_ratio,
        timing_warning=control_ratio > 1.15,
        selections=selections,
    )
    return records, dict(**metadata, measurements=order_records), diagnostic


def _report(payload):
    env = payload["environment"]
    lines = [
        "# Fused add RMSNorm strategy benchmark",
        "",
        f"GPU: {env['gpu']}; PyTorch: {env['torch']}; Triton: {env['triton']}.",
        f"Complete: {payload['complete']}; measurements: "
        f"{len(payload['results'])}/{payload['case_plan']['measurement_count']}.",
        f"Rounds: {payload['config']['rounds']}; warmup per round: "
        f"{payload['config']['warmup']}; repetitions per round: "
        f"{payload['config']['repeat']}; graph unroll: {payload['config']['graph_unroll']}.",
        "",
        "Public eager calls include allocation/autograd and host dispatch gaps. "
        "CUDA Graph diagnostics amortize host launches; compare providers within the same scope "
        "and timing mode. No torch.compile. Forward uses no_grad, backward reuses a graph, "
        "forward+backward builds a fresh graph. Weight reduction uses torch.sum as its baseline, "
        "identical FP32 inputs, and preallocated outputs/parallel workspaces.",
        "",
        "Provider order alternates in paired reversed rounds. Python GC is disabled only during "
        "measurement and restored afterwards. The median is the median of per-round medians. "
        "JSON retains round order, samples, sample standard deviation, wall time and spread. "
        "Graph samples are divided by the captured unroll count. Inputs remain cached; no explicit "
        "cache eviction. GPU graph time is not an end-to-end eager speedup.",
        "",
        "Reported reductions passed FP64 accuracy and repeatability checks. Reported public "
        "calls passed output/gradient accuracy and training/inference byte checks before timing. "
        "Changing reduction configuration may change grad_weight bytes. These checks do not prove "
        "cross-strategy/microbatch/distributed equality. Tuned settings are per-case candidates, "
        "not a production selection rule.",
        "",
        "| Input | Shape | Scope | Timing | Provider | Median ms | Speedup | "
        "Extra peak MiB | Partial workspace MiB |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    native = {
        (r["input_dtype"], tuple(r["shape"]), r["mode"], r["timing"]): r["median_ms"]
        for r in payload["results"]
        if r["provider"] == "native"
    }
    for r in payload["results"]:
        key = (r["input_dtype"], tuple(r["shape"]), r["mode"], r["timing"])
        speedup = native.get(key, r["median_ms"]) / r["median_ms"]
        memory = f"{r['extra_peak_mib']:.2f}" if "extra_peak_mib" in r else "—"
        workspace = f"{r['workspace_mib']:.3f}" if "workspace_mib" in r else "—"
        lines.append(
            f"| {r['input_dtype']} | {r['shape'][0]}x{r['shape'][1]} | {r['mode']} | "
            f"{r['timing']} | {r['provider']} | {r['median_ms']:.6f} | {speedup:.2f}x | "
            f"{memory} | {workspace} |"
        )
    lines.extend(["", "## Timing controls", ""])
    for d in payload.get("diagnostics", []):
        status = "RECHECK before dispatch decisions" if d["timing_warning"] else "within 15%"
        lines.append(
            f"- {d['input_dtype']} {d['shape']}: identical-forward timing spread "
            f"{d['same_forward_spread']:.2f}x — {status}."
        )
        if d["selections"]:
            choices = ", ".join(f"{k}: {v['name']}" for k, v in d["selections"].items())
            lines.append(
                f"  Reduction-graph shortlist, also timed as public `*_tuned` calls: {choices}."
            )
    return "\n".join(lines) + "\n"


def _save(payload, folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (folder / "report.md").write_text(_report(payload), encoding="utf-8")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=_PRESETS, default="model")
    parser.add_argument("--rows", nargs="+", type=_positive_int)
    parser.add_argument("--cols", nargs="+", type=_positive_int)
    parser.add_argument("--dtypes", nargs="+", choices=_DTYPES, default=list(_DTYPES))
    parser.add_argument("--sweep-configs", action="store_true")
    parser.add_argument("--rounds", type=_positive_int, default=4)
    parser.add_argument("--warmup", type=_positive_int, default=10)
    parser.add_argument("--repeat", type=_positive_int, default=50)
    parser.add_argument("--graph-unroll", type=_positive_int, default=16)
    parser.add_argument("--seed", type=int, default=434)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/fused-add-rmsnorm"))
    parser.add_argument("--dry-run", action="store_true", help="Print the case plan without a GPU")
    args = parser.parse_args(argv)
    rows, cols = _PRESETS[args.preset]
    args.rows = list(dict.fromkeys(args.rows or rows))
    args.cols = list(dict.fromkeys(args.cols or cols))
    args.dtypes = list(dict.fromkeys(args.dtypes))
    return args


def _case_plan(args):
    count = len(args.dtypes) * len(args.rows) * len(args.cols)
    reduction_providers = len(_variants(args.sweep_configs)) + 1
    public_providers = 7 if args.sweep_configs else 4
    return dict(
        case_count=count,
        reduction_config_count=reduction_providers - 1,
        measurement_count=count * (reduction_providers * 2 + public_providers * 3 + 2),
    )


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

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload = dict(
        complete=False,
        environment=dict(
            gpu=properties.name,
            capability=list(torch.cuda.get_device_capability()),
            multiprocessor_count=properties.multi_processor_count,
            total_memory=properties.total_memory,
            torch=torch.__version__,
            triton=triton.__version__,
            cuda=torch.version.cuda,
            hip=torch.version.hip,
        ),
        config={**vars(args), "output_dir": str(args.output_dir)},
        case_plan=plan,
        results=[],
        round_orders=[],
        diagnostics=[],
    )
    _save(payload, args.output_dir)
    case_index = 0
    for dtype in args.dtypes:
        for n_rows in args.rows:
            for n_cols in args.cols:
                case_index += 1
                print(
                    f"Case {case_index}/{plan['case_count']}: "
                    f"checking and timing {dtype} [{n_rows}, {n_cols}]...",
                    flush=True,
                )
                records, orders, diagnostic = _run_case(args, module, n_rows, n_cols, dtype)
                payload["results"].extend(records)
                payload["round_orders"].append(orders)
                payload["diagnostics"].append(diagnostic)
                _save(payload, args.output_dir)
                print(
                    f"Same-forward spread: {diagnostic['same_forward_spread']:.2f}x; "
                    f"timing warning: {diagnostic['timing_warning']}",
                    flush=True,
                )
    if len(payload["results"]) != plan["measurement_count"]:
        raise RuntimeError("Incomplete measurement plan")
    payload["complete"] = True
    _save(payload, args.output_dir)
    print(f"Reports saved to {args.output_dir}")


if __name__ == "__main__":
    main()
