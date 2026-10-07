# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark CLI/report checks; performance measurements stay in benchmarks/."""

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
    assert json.loads(capsys.readouterr().out) == {"case_count": 36, "measurement_count": 432}
    assert not target.exists()


@pytest.mark.parametrize("argument", ["--rows", "--cols", "--warmup", "--repeat"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_nonpositive_values(benchmark, argument, value):
    with pytest.raises(SystemExit) as error:
        benchmark._parse_args([argument, value])
    assert error.value.code == 2


def test_cli_default_directory_and_custom_case_plan(benchmark):
    args = benchmark._parse_args(["--rows", "3", "--cols", "129", "--dtypes", "bf16"])
    assert args.output_dir == Path("reports/fused-add-rmsnorm")
    assert benchmark._case_plan(args) == {"case_count": 1, "measurement_count": 12}


def test_no_gpu_fails_before_writing_results(benchmark, monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="CUDA or ROCm"):
        benchmark.main(["--output-dir", str(tmp_path / "missing")])
    assert not (tmp_path / "missing").exists()


def test_reports_preserve_errors_samples_and_native_speedup(benchmark, tmp_path):
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
    for mode in ("forward", "backward", "forward_backward", "weight_reduction"):
        for provider, latency in (("native", 6.0), ("sequential", 3.0), ("tiled", 2.0)):
            payload["results"].append(
                {
                    "input_dtype": "bf16",
                    "shape": [3, 129],
                    "provider": provider,
                    "mode": mode,
                    "median_ms": latency,
                    "std_ms": 0.0,
                    "samples_ms": [latency, latency],
                    "extra_peak_mib": 1.0,
                    "max_abs_error_vs_fp64": 1e-6,
                }
            )
    payload["complete"] = True
    benchmark._save(payload, tmp_path)
    assert json.loads((tmp_path / "results.json").read_text()) == payload
    report = (tmp_path / "report.md").read_text()
    assert "measurements: 12/12" in report
    assert report.count("| 2.00x | 3.00x | 1.50x |") == 4
    assert "torch.sum, not a full operator call" in report


def test_accuracy_check_catches_wrong_gradients(benchmark):
    with pytest.raises(AssertionError):
        benchmark._check(torch.zeros(3), torch.ones(3))
