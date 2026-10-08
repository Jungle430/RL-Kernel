# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark CLI/report checks; performance measurements stay in benchmarks/."""

import gc
import importlib.util
import json
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
    "argument", ["--rows", "--cols", "--warmup", "--repeat", "--rounds", "--graph-unroll"]
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
                    max_abs_error_vs_fp64=1e-6,
                )
            )
    benchmark._save(payload, tmp_path)
    assert json.loads((tmp_path / "results.json").read_text()) == payload
    report = (tmp_path / "report.md").read_text()
    assert "measurements: 6/22" in report
    assert "| eager | parallel | 4.000000 | 2.00x |" in report
    assert "| graph | parallel | 1.000000 | 3.00x |" in report
    assert "| graph | parallel | 1.000000 | 5.00x |" in report
    assert "torch.sum" in report
    assert "not an end-to-end eager speedup" in report


def test_accuracy_check_catches_wrong_gradients(benchmark):
    with pytest.raises(AssertionError):
        benchmark._check(torch.zeros(3), torch.ones(3))
