# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU checks for the shared selection benchmark's shapes and accuracy gates."""

from functools import partial

import pytest
import torch

from benchmarks.operators.embedding import benchmark_index_select as benchmark
from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.reference.embedding.index_select import NativeIndexSelectOp


@pytest.mark.parametrize(
    "shape,dim", [((2, 17, 32), 1), ((17, 32), 0), ((17, 4, 8), 0), ((2, 17), -1)]
)
def test_inputs_and_independent_checks(shape, dim):
    x, index, grad = benchmark.make_inputs(shape, dim, 0.25, torch.float32, "cpu", 680)
    assert x.shape == shape
    assert grad.shape == x.index_select(dim, index).shape
    assert index.numel() == shape[dim] // 2
    assert index.unique().numel() < index.numel()
    reference = NativeIndexSelectOp()
    candidate = partial(reference, dim=dim)
    gold = benchmark.run_vjp(partial(reference.forward_fp32, dim=dim), x, index, grad)[1]
    ordered_dx = benchmark.run_vjp(candidate, x, index, grad)[1]
    tolerance = resolve_tolerance(
        load_contract(), judgment="gradient_accuracy", op_class="reduction", dtype=x.dtype
    )
    checks = benchmark.check_result(
        candidate, x, index, grad, gold, tolerance, dim=dim, ordered_dx=ordered_dx
    )
    assert not benchmark.checks_failed(checks)
    # A repeatable, numerically close result must still fail the exact-order gate.
    checks = benchmark.check_result(
        candidate, x, index, grad, gold, tolerance, dim=dim, ordered_dx=ordered_dx + 1
    )
    assert checks["gradient_accuracy_failures"] == 0
    assert checks["ordered_gradient_bit_mismatches"] > 0
    assert benchmark.checks_failed(checks)


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--dim", "2", "--shapes", "7x9"],
        ["--dim", "0", "--shapes", "0x9"],
        ["--dim", "0", "--duplicates", "1"],
        ["--dim", "0", "--samples", "0"],
    ],
)
def test_cli_rejects_invalid_cases(arguments):
    with pytest.raises(SystemExit):
        benchmark.parse_args(arguments)


def test_dry_run_without_cuda(monkeypatch, capsys):
    import json

    monkeypatch.setattr(
        "sys.argv", ["benchmark_index_select", "--dim", "0", "--shapes", "17x4x8", "--dry-run"]
    )
    benchmark.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["cases"] == [
        {"shape": [17, 4, 8], "dim": 0, "duplicate_fraction": fraction} for fraction in (0.0, 0.25)
    ]
    assert plan["measurement_count"] == 18
