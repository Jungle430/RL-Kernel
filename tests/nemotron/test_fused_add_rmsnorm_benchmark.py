# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark CLI/report checks; performance measurements stay in benchmarks/."""

import gc
import importlib
import importlib.util
import json
import statistics
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def benchmark():
    path = Path(__file__).resolve().parents[2] / "benchmarks/benchmark_fused_add_rmsnorm.py"
    spec = importlib.util.spec_from_file_location("fused_add_rmsnorm_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_requires_no_gpu_and_writes_no_reports(benchmark, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    target = tmp_path / "reports"
    benchmark.main(["--dry-run", "--output-dir", str(target)])
    assert json.loads(capsys.readouterr().out) == {
        "case_count": 36,
        "reduction_config_count": 3,
        "measurement_count": 792,
    }
    assert not target.exists()


@pytest.mark.parametrize(
    "argument",
    ["--rows", "--cols", "--warmup", "--repeat", "--rounds", "--graph-unroll"],
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_nonpositive_values(benchmark, argument, value):
    with pytest.raises(SystemExit) as error:
        benchmark._parse_args([argument, value])
    assert error.value.code == 2


def test_cli_default_directory_and_deduplicated_case_plan(benchmark):
    args = benchmark._parse_args(
        ["--rows", "3", "3", "--cols", "129", "129", "--dtypes", "bf16", "bf16"]
    )
    assert args.output_dir == Path("reports/fused-add-rmsnorm")
    assert benchmark._case_plan(args) == {
        "case_count": 1,
        "reduction_config_count": 3,
        "measurement_count": 22,
    }


def test_tuning_grid_covers_each_strategy_and_keeps_defaults(benchmark):
    specs = benchmark._variants(True)
    assert specs[:3] == benchmark._variants(False)
    keys = [(s["strategy"], s["block_rows"], s["block_cols"], s["num_warps"]) for s in specs]
    assert len(set(keys)) == len(specs) == 36
    assert len({s["name"] for s in specs}) == 36
    assert {s["strategy"] for s in specs} == {"sequential", "tiled", "parallel"}
    assert {s["num_warps"] for s in specs} == {4, 8}
    args = benchmark._parse_args(["--preset", "tuning", "--sweep-configs"])
    assert {31, 32, 33, 8192, 32768} <= set(args.rows)
    assert {128, 129, 2688} <= set(args.cols)
    assert benchmark._case_plan(args) == {
        "case_count": 132,
        "reduction_config_count": 36,
        "measurement_count": 12804,
    }


def test_boundary_plan_uses_fixed_shortlist_for_every_public_scope(benchmark):
    args = benchmark._parse_args(["--preset", "boundary", "--shortlist-configs", "--public-graph"])
    assert args.cols == [2688]
    assert {1, 2, 4, 8, 15, 16, 17, 128, 129, 255, 256, 257, 511, 512, 513, 32768} <= set(args.rows)
    specs = benchmark._variants(False, True)
    assert specs[:3] == benchmark._variants(False)

    def keys(variants):
        return {(s["strategy"], s["block_rows"], s["block_cols"], s["num_warps"]) for s in variants}

    assert len(keys(specs)) == len(specs) == 10
    assert keys(specs) <= keys(benchmark._variants(True))
    assert benchmark._case_plan(args) == {
        "case_count": 75,
        "reduction_config_count": 10,
        "measurement_count": 6600,
    }
    with pytest.raises(SystemExit):
        benchmark._parse_args(["--sweep-configs", "--shortlist-configs"])


def test_dispatch_plan_covers_crossover_tails_and_outside_previous_domain(benchmark):
    args = benchmark._parse_args(["--preset", "dispatch", "--finalist-configs", "--public-graph"])
    shapes = {tuple(shape) for shape in args.shapes}
    # Both sides of all remaining plausible eager cutoffs, not only round sizes.
    for center in (2048, 3072, 4096, 6144, 8192):
        assert {(center + delta, 2688) for delta in (-1, 0, 1)} <= shapes
    assert {
        (1, 2688),
        (1536, 2688),
        (2560, 2688),
        (3584, 2688),
        (5120, 2688),
        (7168, 2688),
        (16384, 2688),
        (32768, 2688),
        (65536, 2688),
    } <= shapes
    for rows in (1, 2048, 8192):
        assert {(rows, width) for width in (129, 2687, 2689, 4096, 8193)} <= shapes
    assert len(shapes) == len(args.shapes) == 50
    assert (
        65536,
        8193,
    ) not in shapes  # Sparse width checks, not an accidental full cross product.
    specs = benchmark._variants(False, finalists=True)
    assert specs[:3] == benchmark._variants(False)
    assert len(specs) == 6

    def keys(variants):
        return {(s["strategy"], s["block_rows"], s["block_cols"], s["num_warps"]) for s in variants}

    assert keys(specs) <= keys(benchmark._variants(False, True))
    plan = benchmark._case_plan(args)
    assert plan["case_count"] == 150
    assert plan["reduction_config_count"] == 6
    assert plan["measurement_count"] == 8400
    assert plan["coverage"] == dict(
        model_width=2688, model_shape_count=35, other_shape_count=15, shapes=args.shapes
    )
    cases = benchmark._cases(args)
    assert len(set(cases)) == len(cases) == 150
    assert {dtype for dtype, _, _ in cases} == {"fp16", "bf16", "fp32"}


def test_dispatch_overrides_and_reverse_run_do_not_expand_or_drop_cases(benchmark):
    options = [
        "--preset",
        "dispatch",
        "--rows",
        "2049",
        "2049",
        "3072",
        "--finalist-configs",
    ]
    args = benchmark._parse_args(options)
    assert args.shapes == [[2049, 2688], [3072, 2688]]
    reverse = benchmark._parse_args(options + ["--reverse-cases"])
    assert benchmark._cases(reverse) == benchmark._cases(args)[::-1]
    explicit = benchmark._parse_args(options + ["--cols", "129", "129", "4097"])
    assert explicit.shapes == [[2049, 129], [2049, 4097], [3072, 129], [3072, 4097]]
    assert benchmark._case_plan(explicit)["case_count"] == 12
    for incompatible in ("--sweep-configs", "--shortlist-configs"):
        with pytest.raises(SystemExit):
            benchmark._parse_args(["--finalist-configs", incompatible])


def _comparison_record(strategy, medians, *, mode="forward_backward", timing="eager"):
    return dict(
        provider=strategy,
        strategy=strategy,
        mode=mode,
        timing=timing,
        median_ms=statistics.median(medians),
        wall_ms_per_call=statistics.median(medians),
        round_spread=max(medians) / min(medians),
        rounds=[dict(index=i, median_ms=value) for i, value in enumerate(medians)],
    )


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([2.0] * 4, [1.0] * 4, "parallel"),
        ([1.0] * 4, [2.0] * 4, "tiled"),
        ([1.02] * 4, [1.0] * 4, "inconclusive"),  # A tiny advantage is not a cutoff.
        ([2.0] * 4, [0.5, 1.0, 1.0, 1.0], "inconclusive"),  # Unstable faster candidate.
        ([0.98, 1.1, 1.1, 1.1], [1.0] * 4, "inconclusive"),  # One round contradicts.
        ([2.0] * 2, [1.0] * 2, "inconclusive"),  # Smoke cannot establish a winner.
    ],
)
def test_strategy_comparison_does_not_promote_noisy_or_tied_minima(
    benchmark, left, right, expected
):
    records = [_comparison_record("tiled", left), _comparison_record("parallel", right)]
    (comparison,) = benchmark._strategy_comparisons(records)
    assert comparison["candidate"] == expected
    assert comparison["round_left_over_right"] == [a / b for a, b in zip(left, right, strict=True)]
    assert comparison["right_wins"] == sum(a > b for a, b in zip(left, right, strict=True))


def test_strategy_comparison_separates_scopes_and_checks_wall_time_and_rounds(
    benchmark,
):
    records = []
    for mode in ("backward", "forward_backward"):
        for timing in ("eager", "graph"):
            faster, slower = ("tiled", "parallel") if timing == "eager" else ("parallel", "tiled")
            records.extend(
                [
                    _comparison_record(faster, [1.0] * 4, mode=mode, timing=timing),
                    _comparison_record(slower, [2.0] * 4, mode=mode, timing=timing),
                ]
            )
    comparisons = benchmark._strategy_comparisons(records)
    assert len(comparisons) == 4
    for c in comparisons:
        assert c["candidate"] == ("tiled" if c["timing"] == "eager" else "parallel")
    # Event timing and instrumented wall time disagree: retain the evidence,
    # but do not recommend a dispatch change based only on the event winner.
    records[0]["wall_ms_per_call"] = 3.0
    assert benchmark._strategy_comparisons(records)[0]["candidate"] == "inconclusive"
    records[0]["rounds"][0]["index"] = 7
    with pytest.raises(ValueError, match="matching round indices"):
        benchmark._strategy_comparisons(records)


def test_captured_return_checks_bits_shape_and_dtype(benchmark):
    benchmark._assert_same_outputs((torch.ones(3),), (torch.ones(3),))
    for actual, expected in (
        (torch.tensor([0.0]), torch.tensor([-0.0])),
        (torch.ones(1, 3), torch.ones(3)),
        (torch.ones(3, dtype=torch.float64), torch.ones(3)),
    ):
        with pytest.raises(AssertionError, match="CUDA Graph"):
            benchmark._assert_same_outputs((actual,), (expected,))


@pytest.mark.parametrize("mode", ["backward", "forward_backward"])
def test_graph_gradient_calls_use_fresh_leaves_while_eager_graph_stays_alive(benchmark, mode):
    inputs = tuple(torch.randn(3, requires_grad=True) for _ in range(3))
    upstream = (torch.randn(3), torch.randn(3))
    observed = []

    def op(x, residual, weight):
        observed.append((x, residual, weight))
        updated = x + residual
        return updated * weight, updated

    eager_outputs = op(*inputs)
    expected = torch.autograd.grad(eager_outputs, inputs, upstream, retain_graph=True)
    call = benchmark._graph_grad_call(op, inputs, upstream, mode)
    for _ in range(2):
        actual = call()
        benchmark._assert_same_outputs(actual, expected)
        for original, fresh in zip(inputs, observed[-1], strict=True):
            assert fresh is not original
            assert fresh.is_leaf and fresh.requires_grad
            assert fresh.data_ptr() == original.data_ptr()
            assert fresh.dtype == original.dtype and fresh.device == original.device
            assert original.grad is None and fresh.grad is None
    # Constructing the capture call must not consume or mutate the old graph.
    benchmark._assert_same_outputs(torch.autograd.grad(eager_outputs, inputs, upstream), expected)


@pytest.mark.parametrize("corrupt_during_timing", [False, True])
def test_graph_results_are_rechecked_after_timed_replays(
    benchmark, monkeypatch, corrupt_during_timing
):
    # A capture can initially be correct but later overwrite its saved state.
    # Exercise the result gate on CPU without substituting for real GPU coverage.
    result = (torch.ones(3),)
    reference = (torch.ones(3),)
    checks = []
    active_stream = None

    class Stream:
        def wait_stream(self, other):
            pass

    capture_stream = Stream()

    @contextmanager
    def use_stream(stream):
        nonlocal active_stream
        previous = active_stream
        active_stream = stream
        try:
            yield
        finally:
            active_stream = previous

    def eager_call():
        assert active_stream is capture_stream
        return reference

    class Graph:
        def replay(self):
            pass

    def capture(fn, unroll, *, stream):
        assert stream is capture_stream
        assert active_stream is capture_stream
        return Graph(), result

    def measure(fn, repeat, divisor):
        fn()
        if corrupt_during_timing:
            result[0].fill_(float("nan"))
        return dict(samples_ms=[1.0, 1.0], median_ms=1.0, wall_ms_per_call=1.0)

    def check(values):
        checks.append(True)
        return {"grad_weight": benchmark._check(values[0], reference[0])}

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: active_stream)
    monkeypatch.setattr(torch.cuda, "stream", use_stream)
    monkeypatch.setattr(benchmark, "_capture", capture)
    monkeypatch.setattr(benchmark, "_measure_block", measure)
    args = benchmark._parse_args(["--rounds", "1", "--warmup", "1", "--repeat", "2"])
    kwargs = dict(timing="graph", check_result=check, capture_stream=capture_stream)
    if corrupt_during_timing:
        with pytest.raises(AssertionError):
            benchmark._measure_group({"test": eager_call}, args, **kwargs)
    else:
        records, _ = benchmark._measure_group({"test": eager_call}, args, **kwargs)
        assert records["test"]["graph_matches_eager_bitwise"]
        assert records["test"]["graph_repeat_bitwise"]
        assert records["test"]["graph_max_abs_errors_vs_native"] == {"grad_weight": 0.0}
    assert len(checks) == 2
    assert active_stream is None


def test_paired_round_orders_balance_positions(benchmark):
    names = ["native", "sequential", "tiled", "parallel"]
    orders = [benchmark._round_order(names, r) for r in range(4)]
    for order in orders:
        assert sorted(order) == sorted(names)
    assert orders[1] == orders[0][::-1]
    assert orders[3] == orders[2][::-1]
    assert orders[0] != orders[2]
    assert {sum(order.index(name) for order in orders) for name in names} == {6}


def test_summary_uses_round_medians_and_keeps_outliers(benchmark):
    rounds = [
        dict(samples_ms=[1.0, 1.0, 100.0], median_ms=1.0, wall_ms_per_call=5.0),
        dict(samples_ms=[3.0, 3.0, 3.0], median_ms=3.0, wall_ms_per_call=7.0),
    ]
    result = benchmark._summarize(rounds)
    assert result["median_ms"] == 2.0
    assert result["round_spread"] == 3.0
    assert result["samples_ms"] == [1.0, 1.0, 100.0, 3.0, 3.0, 3.0]
    assert result["std_ms"] > 30.0
    assert result["wall_ms_per_call"] == 6.0
    assert result["rounds"] == rounds


@pytest.mark.parametrize("initially_enabled", [False, True])
def test_timing_failure_restores_gc_state(benchmark, monkeypatch, initially_enabled):
    original = gc.isenabled()
    gc.enable() if initially_enabled else gc.disable()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    args = benchmark._parse_args(["--warmup", "1", "--repeat", "1"])

    def fail(*args):
        assert not gc.isenabled()
        raise RuntimeError("timer failed")

    monkeypatch.setattr(benchmark, "_measure_block", fail)
    try:
        with pytest.raises(RuntimeError, match="timer failed"):
            benchmark._measure_group({"test": lambda: None}, args, timing="eager")
        assert gc.isenabled() == initially_enabled
    finally:
        gc.enable() if original else gc.disable()


def test_no_gpu_fails_before_writing_results(benchmark, monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="CUDA or ROCm"):
        benchmark.main(["--output-dir", str(tmp_path / "missing")])
    assert not (tmp_path / "missing").exists()


def test_report_does_not_mix_eager_graph_or_kernel_baselines(benchmark, tmp_path):
    args = benchmark._parse_args(["--rows", "3", "--cols", "129", "--dtypes", "bf16"])
    payload = {
        "complete": False,
        "environment": {"gpu": "test GPU", "torch": "test", "triton": "test"},
        "config": {**vars(args), "output_dir": str(tmp_path)},
        "case_plan": benchmark._case_plan(args),
        "results": [],
    }
    benchmark._save(payload, tmp_path)
    assert not json.loads((tmp_path / "results.json").read_text())["complete"]
    # Each scope has a different baseline: mixing them would produce wrong speedups.
    for timing, mode, latencies in (
        ("eager", "forward", (8.0, 4.0)),
        ("graph", "forward", (3.0, 1.0)),
        ("graph", "weight_reduction", (5.0, 1.0)),
        ("eager", "backward", (6.0, 2.0)),
        ("graph", "backward", (4.0, 1.0)),
        ("graph", "forward_backward", (7.0, 1.0)),
    ):
        for provider, latency in zip(("native", "parallel"), latencies, strict=True):
            payload["results"].append(
                dict(
                    input_dtype="bf16",
                    shape=[3, 129],
                    provider=provider,
                    mode=mode,
                    timing=timing,
                    median_ms=latency,
                    samples_ms=[latency, latency],
                    round_spread=1.5,
                    max_abs_error_vs_fp64=1e-6,
                )
            )
    payload["diagnostics"] = [
        dict(
            input_dtype="bf16",
            shape=[3, 129],
            same_forward_spread=1.01,
            timing_warning=True,
            selections={},
            unstable_measurements=[
                dict(mode="backward", timing="eager", provider="parallel", round_spread=1.5),
            ],
        )
    ]
    benchmark._save(payload, tmp_path)
    assert json.loads((tmp_path / "results.json").read_text()) == payload
    report = (tmp_path / "report.md").read_text()
    assert "measurements: 12/22" in report
    assert "| eager | parallel | 4.000000 | 2.00x |" in report
    assert "| graph | parallel | 1.000000 | 3.00x |" in report
    assert "| graph | parallel | 1.000000 | 5.00x |" in report
    assert "| backward | graph | parallel | 1.000000 | 4.00x |" in report
    assert "| forward_backward | graph | parallel | 1.000000 | 7.00x |" in report
    assert "Round max/min" in report
    assert "within 15%" in report
    assert "RECHECK 1 eager measurements" in report
    assert "torch.sum" in report
    assert "not an end-to-end eager speedup" in report


def test_accuracy_check_catches_wrong_gradients(benchmark):
    with pytest.raises(AssertionError):
        benchmark._check(torch.zeros(3), torch.ones(3))


def test_report_keeps_strategy_comparison_timing_and_candidate_status(benchmark):
    args = benchmark._parse_args(["--rows", "3072", "--cols", "2688", "--dtypes", "bf16"])
    records = []
    for timing, left, right in (("eager", 1.02, 1.0), ("graph", 2.0, 1.0)):
        for strategy, value in (("tiled", left), ("parallel", right)):
            records.append(
                {
                    **_comparison_record(strategy, [value] * 4, timing=timing),
                    "input_dtype": "bf16",
                    "shape": [3072, 2688],
                }
            )
    comparisons = benchmark._strategy_comparisons(records)
    payload = dict(
        complete=False,
        environment=dict(gpu="test", torch="test", triton="test"),
        config=vars(args),
        case_plan=benchmark._case_plan(args),
        results=records,
        diagnostics=[
            dict(
                input_dtype="bf16",
                shape=[3072, 2688],
                same_forward_spread=1.0,
                selections={},
                strategy_comparisons=comparisons,
            )
        ],
    )
    report = benchmark._report(payload)
    assert "| eager | tiled | parallel | 1.020x | 1.020x | 4/4 | inconclusive |" in report
    assert "| graph | tiled | parallel | 2.000x | 2.000x | 4/4 | parallel |" in report
    assert "not confidence intervals" in report


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU required")
@pytest.mark.filterwarnings("error:.*AccumulateGrad.*:UserWarning")
@pytest.mark.parametrize(
    "configs,expected_records,expected_graph",
    [("--shortlist-configs", 88, 33), ("--finalist-configs", 56, 21)],
)
def test_public_graph_shortlist_matches_eager_and_reference_after_replays(
    benchmark, configs, expected_records, expected_graph
):
    triton = pytest.importorskip("triton")
    if not hasattr(triton, "jit"):
        pytest.skip("A working Triton runtime is required")
    module = importlib.import_module("rl_engine.kernels.ops.triton.norm.fused_add_rmsnorm")
    args = benchmark._parse_args(
        [
            "--rows",
            "257",
            "--cols",
            "129",
            "--dtypes",
            "bf16",
            configs,
            "--public-graph",
            "--rounds",
            "1",
            "--warmup",
            "1",
            "--repeat",
            "2",
            "--graph-unroll",
            "2",
        ]
    )
    records, _, _ = benchmark._run_case(args, module, 257, 129, "bf16")
    assert len(records) == benchmark._case_plan(args)["measurement_count"] == expected_records
    graph_results = [
        r for r in records if r["timing"] == "graph" and r["mode"] != "weight_reduction"
    ]
    assert len(graph_results) == expected_graph
    for r in graph_results:
        assert r["graph_matches_eager_bitwise"]
        assert r["graph_repeat_bitwise"]
        labels = (
            {"y", "updated_residual"}
            if r["mode"] == "forward"
            else {"grad_x", "grad_residual", "grad_weight"}
        )
        assert set(r["graph_max_abs_errors_vs_native"]) == labels
    # The smoke's single round cannot establish a performance recommendation.
    assert len(benchmark._strategy_comparisons(records)) == 8
    assert all(c["candidate"] == "inconclusive" for c in benchmark._strategy_comparisons(records))
