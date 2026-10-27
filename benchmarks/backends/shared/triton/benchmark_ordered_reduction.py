# SPDX-License-Identifier: Apache-2.0
"""Bounded torch.sum / tl.sum / ordered-sum comparison and scheduling sweep.

Run as a module from the repository root. Each shape/dtype runs in an isolated
process with a wall-clock timeout covering initialization, compilation, accuracy
and timing. CPU hosts can inspect --dry-run and run the benchmark's unit tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.contracts.operators.ordered_reduction import MAX_GROUP_SIZE, TILE_SIZE
from rl_engine.reference import reduction as reference

MODULE = "benchmarks.backends.shared.triton.benchmark_ordered_reduction"
ROOT = Path(__file__).resolve().parents[4]
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
DEFAULT_SHAPES = [(1, 1024), (16, 1025), (128, 2688), (1024, 4096), (16, 262144), (1024, 262144)]
ROW_LIMIT = 32768
"""Bound full-row tl.sum compilation; the two-stage baseline covers larger lengths."""


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def power_of_two(value):
    number = positive(value)
    if number & (number - 1):
        raise argparse.ArgumentTypeError("must be a power of two")
    return number


def shape(value):
    try:
        rows, length = value.lower().split("x")
        return positive(rows), positive(length)
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError("expected positive MxN, such as 16x1025") from error


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--shapes", type=shape, nargs="+", default=DEFAULT_SHAPES)
    cli.add_argument("--dtypes", choices=DTYPES, nargs="+", default=list(DTYPES))
    cli.add_argument("--timings", choices=("eager", "graph"), nargs="+", default=["eager", "graph"])
    cli.add_argument("--warps", type=int, choices=(4, 8), nargs="+", default=[4])
    cli.add_argument(
        "--pipeline-stages", type=int, choices=(0, 1, 2, 3, 4), nargs="+", default=[0, 3]
    )
    cli.add_argument("--merge-unroll", type=positive, nargs="+", default=[1])
    cli.add_argument("--block-size", type=power_of_two, default=TILE_SIZE)
    cli.add_argument("--max-group-size", type=power_of_two, default=MAX_GROUP_SIZE)
    cli.add_argument(
        "--padding", type=int, default=0, help="physical suffix columns, filled with NaN"
    )
    cli.add_argument(
        "--distribution",
        choices=("uniform", "normal"),
        default="uniform",
        help="uniform [0, 1) for throughput; normal for cancellation stress",
    )
    cli.add_argument("--device", type=int, default=0)
    cli.add_argument("--rounds", type=positive, default=3)
    cli.add_argument("--warmup", type=positive, default=10)
    cli.add_argument("--repeat", type=positive, default=30)
    cli.add_argument("--graph-calls", type=positive, default=16)
    cli.add_argument(
        "--case-timeout", type=positive, default=180, help="seconds, including compilation"
    )
    cli.add_argument("--seed", type=int, default=939)
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument(
        "--keep-going", action="store_true", help="continue after a failed/timed-out case"
    )
    cli.add_argument(
        "--dry-run", action="store_true", help="print the exact plan without GPU access"
    )
    return cli


def providers(args, length):
    """Scheduling candidates share one tree; native baselines have independent trees."""
    result = [dict(name="torch.sum", family="torch")]
    for warps in dict.fromkeys(args.warps):
        for family in ("tl_row", "tl_parallel", "ordered_serial", "ordered_parallel"):
            if family == "tl_row" and length > ROW_LIMIT:
                continue
            pipelines = (
                list(dict.fromkeys(args.pipeline_stages)) if family == "ordered_serial" else [0]
            )
            unrolls = (
                list(dict.fromkeys(args.merge_unroll)) if family == "ordered_parallel" else [1]
            )
            for pipeline in pipelines:
                for unroll in unrolls:
                    name = f"{family}/w{warps}"
                    if family == "ordered_serial":
                        name += f"/p{pipeline}"
                    if family == "ordered_parallel":
                        name += f"/u{unroll}"
                    result.append(
                        dict(
                            name=name,
                            family=family,
                            warps=warps,
                            pipeline=pipeline or None,
                            merge_unroll=unroll,
                        )
                    )
    return result


def plan(args):
    if args.padding < 0 or args.device < 0:
        raise ValueError("padding and device must be nonnegative")
    if args.block_size > ROW_LIMIT:
        raise ValueError(f"benchmark tile size is capped at {ROW_LIMIT} to bound compilation")
    tree = dict(
        block_size=args.block_size,
        max_group_size=args.max_group_size,
    )
    cases = []
    for dtype in dict.fromkeys(args.dtypes):
        for rows, length in dict.fromkeys(args.shapes):
            cases.append(
                dict(
                    dtype=dtype,
                    rows=rows,
                    length=length,
                    padding=args.padding,
                    tree=tree,
                    distribution=args.distribution,
                    providers=providers(args, length),
                    timings=list(dict.fromkeys(args.timings)),
                    rounds=args.rounds,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    graph_calls=args.graph_calls,
                    seed=args.seed,
                    device=args.device,
                    omitted=["tl_row: full-row compilation limit"] if length > ROW_LIMIT else [],
                )
            )
    return dict(
        cases=cases,
        case_count=len(cases),
        measurement_count=sum(len(case["providers"]) * len(case["timings"]) for case in cases),
        case_timeout_seconds=args.case_timeout,
    )


def write_json(path, value):
    """Publish checkpoints atomically so timeout recovery never reads half a JSON file."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def accuracy(actual, gold, tolerance):
    """Independent FP64 oracle, with the repository's FP32 reduction tolerances."""
    actual = actual.detach().cpu().double()
    gold = gold.detach().cpu().double()
    error = (actual - gold).abs()
    allowed = tolerance.atol + tolerance.rtol * gold.abs()
    passed = torch.isfinite(actual) & torch.isfinite(gold) & (error <= allowed)
    return dict(
        passed=bool(passed.all()),
        failures=int((~passed).sum()),
        max_abs_error=float(error.max()) if torch.isfinite(error).all() else None,
    )


def summarize(rounds):
    samples = [sample for values in rounds for sample in values]
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("timing samples must be finite and positive")
    medians = [statistics.median(values) for values in rounds]
    return dict(
        median_ms=statistics.median(medians),
        round_medians_ms=medians,
        stdev_ms=statistics.stdev(samples) if len(samples) > 1 else 0.0,
        round_spread=max(medians) / min(medians),
        samples_ms=rounds,
    )


def measure(call, warmup, repeat, divisor):
    """Eager measures one invocation; graph samples replay a captured batch / divisor."""
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repeat)
    ]
    for start, end in events:
        start.record()
        call()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) / divisor for start, end in events]


def run_case(spec, path):
    """Validate all candidates before interleaved timing; checkpoint each phase."""
    import triton

    from benchmarks.backends.shared.triton.ordered_reduction_kernels import make_call

    report = dict(spec=spec, complete=False, phase="setup", results=[])

    def checkpoint(phase):
        report["phase"] = phase
        print(phase, flush=True)
        write_json(path, report)

    checkpoint("setup")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA/ROCm PyTorch environment and GPU are required")
        torch.cuda.set_device(spec["device"])
        device = torch.device("cuda", spec["device"])
        report["environment"] = dict(
            python=sys.version,
            torch=torch.__version__,
            triton=triton.__version__,
            cuda=torch.version.cuda,
            hip=torch.version.hip,
            gpu=torch.cuda.get_device_name(device),
        )
        properties = torch.cuda.get_device_properties(device)
        report["environment"].update(
            capability=torch.cuda.get_device_capability(device),
            multiprocessors=properties.multi_processor_count,
            total_memory_bytes=properties.total_memory,
            compiler_options=dict(
                enable_fp_fusion=False,
                **({"allow_flush_denorm": False} if torch.version.hip else {}),
            ),
        )
        rows, length = spec["rows"], spec["length"]
        dtype = DTYPES[spec["dtype"]]
        input_bytes = (
            rows * (length + spec["padding"]) * torch.empty((), dtype=dtype).element_size()
        )
        if input_bytes > torch.cuda.mem_get_info(device)[0] // 3:
            raise RuntimeError("input exceeds one third of free GPU memory; use a smaller shape")
        generator = torch.Generator(device=device).manual_seed(spec["seed"])
        generate = torch.rand if spec["distribution"] == "uniform" else torch.randn
        x = generate(
            (rows, length + spec["padding"]), device=device, dtype=dtype, generator=generator
        )
        x[:, length:] = float("nan")
        with torch.no_grad():
            gold = torch.sum(x[:, :length], dim=-1, dtype=torch.float64).cpu()
            sampled_rows = sorted({0, rows // 2, rows - 1})
            fixed = reference.sum(
                x[sampled_rows, :length].cpu(),
                from_memory=True,
                compute_dtype=torch.float32,
                **spec["tree"],
            )
            tolerance = resolve_tolerance(
                load_contract(), judgment="forward_accuracy", op_class="reduction", dtype="float32"
            )
            report["tolerance"] = tolerance.to_dict()
            report["fixed_reference_rows"] = sampled_rows
            calls, records, canonical = {}, {}, None
            for provider in spec["providers"]:
                name = provider["name"]
                checkpoint(f"compile/check {name}")
                out = torch.empty(rows, device=device, dtype=torch.float32)
                call, scratch_bytes = make_call(x, out, length, provider, spec["tree"])
                call()
                torch.cuda.synchronize()
                actual = out.cpu()
                check = accuracy(actual, gold, tolerance)
                if not check["passed"]:
                    report["failed_validation"] = dict(provider=name, accuracy=check)
                    raise AssertionError(f"{name}: independent FP64 accuracy check failed")
                if provider["family"].startswith("ordered_"):
                    if not torch.equal(
                        actual[sampled_rows].view(torch.int32), fixed.view(torch.int32)
                    ):
                        raise AssertionError(f"{name}: fixed-order PyTorch reference mismatch")
                    if canonical is not None and not torch.equal(
                        actual.view(torch.int32), canonical
                    ):
                        raise AssertionError(f"{name}: cross-schedule bitwise mismatch")
                    canonical = actual.view(torch.int32)
                torch.cuda.reset_peak_memory_stats()
                allocated = torch.cuda.memory_allocated()
                call()
                torch.cuda.synchronize()
                extra_peak = torch.cuda.max_memory_allocated() - allocated
                for timing in spec["timings"]:
                    timed_call, divisor = call, 1
                    if timing == "graph":
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            for _ in range(spec["graph_calls"]):
                                call()
                        timed_call, divisor = graph.replay, spec["graph_calls"]
                        graph.replay()
                        torch.cuda.synchronize()
                        if not torch.equal(out.cpu().view(torch.int32), actual.view(torch.int32)):
                            raise AssertionError(f"{name}: graph replay changed output bits")
                    key = (name, timing)
                    # A graph stores addresses, not ownership of the original
                    # tensors. Retain the callable owning out/partials even
                    # when only graph timing was requested.
                    calls[key] = (timed_call, divisor, call)
                    records[key] = dict(
                        provider=provider,
                        timing=timing,
                        accuracy=check,
                        scratch_bytes=scratch_bytes,
                        extra_peak_bytes=extra_peak,
                        rounds=[],
                    )
            keys = list(calls)
            for round_index in range(spec["rounds"]):
                random.Random(spec["seed"] + round_index).shuffle(keys)
                for key in keys:
                    checkpoint(f"timing round {round_index + 1}: {key[0]} {key[1]}")
                    call, divisor, _buffer_owner = calls[key]
                    record = records[key]
                    rounds = record["rounds"] + [
                        measure(call, spec["warmup"], spec["repeat"], divisor)
                    ]
                    record.update(rounds=rounds, **summarize(rounds))
                    record["effective_gb_s"] = (rows * length * x.element_size() + rows * 4) / (
                        record["median_ms"] * 1e6
                    )
                    report["results"] = list(records.values())
            report["complete"] = True
            checkpoint("complete")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        write_json(path, report)
        raise


def run_isolated(command, log, timeout):
    """Bound the complete worker, including compiler subprocesses, on POSIX hosts."""
    with log.open("w") as output:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Also stop the compiler group when the user interrupts the runner.
            if process.poll() is None:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass  # The worker exited between poll() and kill().
                process.wait()
        return process.returncode, timed_out


def provenance():
    files = [
        Path(__file__),
        Path(__file__).with_name("ordered_reduction_kernels.py"),
        ROOT / "rl_engine/backends/shared/triton/reduction/ordered_reduce.py",
        ROOT / "rl_engine/reference/reduction/ordered_reduce.py",
        ROOT / "rl_engine/contracts/operators/ordered_reduction.py",
        ROOT / "rl_engine/contracts/profiles/precision/ws1.json",
    ]
    return dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        worktree=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    )


def markdown(report):
    lines = [
        "# Ordered sum benchmark",
        "",
        f"Complete: {report['complete']}",
        "",
        f"Source commit: `{report['source']['commit']}`; "
        f"dirty worktree: {bool(report['source']['worktree'])}.",
        "",
        "FP16/BF16/FP32 inputs; all providers accumulate and output FP32. "
        "Eager torch.sum(out=...) and native tl.sum are independent-tree performance baselines.",
        "Outputs and partial buffers are preallocated. Compilation, validation and graph capture "
        "are excluded. Inputs are reused without cache eviction. Eager event latency can include "
        "host dispatch gaps; graph samples replay a batch and report time per invocation. "
        "Timing modes must be compared separately.",
        "",
        "Speedups divide baseline latency by candidate latency. Best tl.sum means the fastest "
        "measured plain row/two-stage configuration in the same case and timing mode. "
        "Effective GB/s counts input plus output bytes only, not partial traffic. Scratch is "
        "preallocated workspace; extra peak measures allocator growth in a warmed call and "
        "excludes inputs, outputs, scratch and graph capture. Raw rounds and errors are in JSON.",
        "",
        "| Input / shape | Provider | Timing | Median ms | vs torch | vs best tl.sum | "
        "GB/s | Scratch MiB | Extra peak MiB | Round spread |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    failures = []
    for case in report["cases"]:
        spec = case["spec"]
        label = f"{spec['dtype']} {spec['rows']}x{spec['length']}"
        if not case.get("complete"):
            failures.append(
                f"**Incomplete {label}:** {case.get('error', case.get('phase', 'unknown'))}"
            )
            continue
        for row in case["results"]:
            comparable = [r for r in case["results"] if r["timing"] == row["timing"]]
            torch_ms = next(
                r["median_ms"] for r in comparable if r["provider"]["family"] == "torch"
            )
            tl_ms = min(
                r["median_ms"] for r in comparable if r["provider"]["family"].startswith("tl_")
            )
            ms = row["median_ms"]
            lines.append(
                f"| {label} | {row['provider']['name']} | {row['timing']} | {ms:.6f} | "
                f"{torch_ms / ms:.2f}x | {tl_ms / ms:.2f}x | {row['effective_gb_s']:.2f} | "
                f"{row['scratch_bytes'] / 2**20:.3f} | {row['extra_peak_bytes'] / 2**20:.3f} | "
                f"{row['round_spread']:.2f}x |"
            )
    lines.extend(
        [
            "",
            *failures,
            "",
            "## Reproduction",
            "",
            "```json",
            json.dumps(
                dict(
                    environment=next(
                        (
                            case.get("environment")
                            for case in report["cases"]
                            if case.get("environment")
                        ),
                        None,
                    ),
                    cases=[
                        {key: value for key, value in case["spec"].items() if key != "providers"}
                        for case in report["cases"]
                    ],
                ),
                indent=2,
            ),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = parser().parse_args(argv)
    case_plan = plan(args)
    if args.dry_run:
        print(json.dumps(case_plan, indent=2))
        return 0
    folder = args.output_dir or Path("artifacts/benchmarks") / datetime.now().strftime(
        "ordered-sum-%Y%m%d-%H%M%S"
    )
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(
        complete=False,
        plan=case_plan,
        source=provenance(),
        cases=[],
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json(folder / "results.json", report)
    for index, spec in enumerate(case_plan["cases"]):
        path = folder / f"case-{index:03d}.json"
        print(
            f"Case {index + 1}/{case_plan['case_count']}: {spec['dtype']} "
            f"{spec['rows']}x{spec['length']} (timeout {args.case_timeout}s)",
            flush=True,
        )
        started = time.monotonic()
        code, timed_out = run_isolated(
            [sys.executable, "-m", MODULE, "--worker", json.dumps(spec), str(path)],
            folder / f"case-{index:03d}.log",
            args.case_timeout,
        )
        case = json.loads(path.read_text()) if path.exists() else dict(spec=spec, complete=False)
        case.update(exit_code=code, timed_out=timed_out, elapsed_seconds=time.monotonic() - started)
        if timed_out:
            case.update(complete=False, error=f"timeout during {case.get('phase', 'startup')}")
        elif code:
            case.update(
                complete=False, error=case.get("error", f"worker exited {code}; see case log")
            )
        report["cases"].append(case)
        write_json(folder / "results.json", report)
        (folder / "report.md").write_text(markdown(report))
        if not case["complete"] and not args.keep_going:
            break
    report["complete"] = len(report["cases"]) == case_plan["case_count"] and all(
        case["complete"] for case in report["cases"]
    )
    write_json(folder / "results.json", report)
    (folder / "report.md").write_text(markdown(report))
    print(f"Reports: {folder}", flush=True)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        try:
            run_case(json.loads(sys.argv[2]), Path(sys.argv[3]))
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    else:
        raise SystemExit(main())
