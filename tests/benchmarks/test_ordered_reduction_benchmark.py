# SPDX-License-Identifier: Apache-2.0
"""Benchmark planning, validation, timeout recovery and GPU smoke coverage."""

import json
import sys
import time

import pytest
import torch

from benchmarks.backends.shared.triton.benchmark_ordered_reduction import (
    MODULE,
    accuracy,
    main,
    markdown,
    parser,
    plan,
    run_isolated,
    summarize,
)
from rl_engine.contracts.numerical import load_contract, resolve_tolerance


def test_dry_run_needs_no_triton_or_gpu(monkeypatch, capsys):
    def unexpected_gpu_query():
        pytest.fail("dry run must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_gpu_query)
    assert main(["--dry-run", "--shapes", "1x1025", "--dtypes", "bf16"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["case_count"] == 1
    assert result["measurement_count"] == 12


def test_scheduling_sweep_preserves_one_tree_and_deduplicates_baselines():
    args = parser().parse_args(
        [
            "--shapes",
            "1x1024",
            "1x262144",
            "--dtypes",
            "fp32",
            "--warps",
            "4",
            "8",
            "--pipeline-stages",
            "0",
            "2",
            "3",
            "3",
            "--merge-unroll",
            "1",
            "3",
            "--block-size",
            "2048",
            "--padding",
            "37",
        ]
    )
    report = plan(args)
    for case in report["cases"]:
        assert case["tree"] == dict(block_size=2048, max_group_size=8)
        assert case["padding"] == 37
        names = [p["name"] for p in case["providers"]]
        assert len(names) == len(set(names))
        assert names.count("torch.sum") == 1
        assert any(p["family"] == "tl_parallel" for p in case["providers"])
    assert any(p["family"] == "tl_row" for p in report["cases"][0]["providers"])
    assert not any(p["family"] == "tl_row" for p in report["cases"][1]["providers"])


@pytest.mark.parametrize(
    "option,value",
    [("--shapes", "0x32"), ("--block-size", "3"), ("--repeat", "0"), ("--case-timeout", "0")],
)
def test_invalid_workloads_are_rejected(option, value):
    with pytest.raises(SystemExit):
        parser().parse_args([option, value])


def test_accuracy_uses_the_shared_contract_and_rejects_nonfinite_outputs():
    tolerance = resolve_tolerance(
        load_contract(), judgment="forward_accuracy", op_class="reduction", dtype="float32"
    )
    gold = torch.tensor([1.0, -1.0], dtype=torch.float64)
    assert accuracy(gold.float(), gold, tolerance)["passed"]
    assert not accuracy(torch.tensor([float("nan"), -1]), gold, tolerance)["passed"]
    assert accuracy(torch.tensor([1.1, -1]), gold, tolerance)["failures"] == 1


def test_round_aggregation_keeps_raw_samples_and_exposes_variation():
    result = summarize([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    assert result["median_ms"] == 3
    assert result["round_spread"] == 2
    assert result["samples_ms"] == [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]
    for samples in ([], [[0.0]], [[float("nan")]]):
        with pytest.raises(ValueError):
            summarize(samples)


def test_worker_timeout_returns_control_and_preserves_log(tmp_path):
    started = time.monotonic()
    log = tmp_path / "worker.log"
    code, timed_out = run_isolated(
        [sys.executable, "-u", "-c", "import time; print('compile'); time.sleep(30)"],
        log,
        1,
    )
    assert code != 0 and timed_out
    assert time.monotonic() - started < 10
    assert "compile" in log.read_text()


def test_markdown_compares_matching_timing_modes_and_hides_incomplete_results():
    rows = []
    for timing, factor in (("eager", 10), ("graph", 1)):
        for family, ms in (("torch", 8), ("tl_row", 4), ("tl_parallel", 2), ("ordered_serial", 1)):
            rows.append(
                dict(
                    provider=dict(family=family, name=family),
                    timing=timing,
                    median_ms=ms * factor,
                    effective_gb_s=100,
                    scratch_bytes=0,
                    extra_peak_bytes=0,
                    round_spread=1,
                )
            )
    case = dict(spec=dict(dtype="fp32", rows=1, length=1024), complete=True, results=rows)
    report = dict(complete=True, cases=[case], source=dict(commit="test", worktree=""))
    text = markdown(report)
    assert "| 8.00x | 2.00x |" in text
    assert "ordered_serial | eager | 10.000000 | 8.00x | 2.00x" in text
    assert "ordered_serial | graph | 1.000000 | 8.00x | 2.00x" in text
    case.update(complete=False, error="accuracy failed")
    text = markdown(report)
    assert "accuracy failed" in text
    assert "8.00x" not in text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm GPU is required")
@pytest.mark.parametrize("timings", (["graph"], ["eager", "graph"]))
def test_gpu_benchmark_smoke_checks_all_providers_and_replays(tmp_path, timings):
    args = parser().parse_args(
        [
            "--shapes",
            "3x1025",
            "--dtypes",
            "bf16",
            "--padding",
            "17",
            "--rounds",
            "1",
            "--repeat",
            "2",
            "--warmup",
            "1",
            "--graph-calls",
            "2",
            "--pipeline-stages",
            "0",
            "--timings",
            *timings,
        ]
    )
    spec = plan(args)["cases"][0]
    path, log = tmp_path / "result.json", tmp_path / "log.txt"
    code, timed_out = run_isolated(
        [sys.executable, "-m", MODULE, "--worker", json.dumps(spec), str(path)],
        log,
        90,
    )
    assert not timed_out and code == 0, log.read_text()
    result = json.loads(path.read_text())
    assert result["complete"]
    assert len(result["results"]) == len(spec["providers"]) * len(timings)
    assert all(row["accuracy"]["passed"] and row["median_ms"] > 0 for row in result["results"])
