"""Measure hot CPU submit latency of the old and new Triton public entry points on one idle GPU."""
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from benchmarks.models.benchmark_h3_mm_rope import LAYOUTS, timed_call
from rl_engine.backends.shared.triton.rope.h3_mm_rope import TritonH3MMRoPEOp
from rl_engine.validation.models.h3_cases import h3_fl2va_layout
from rl_engine.validation.models.h3_provider import provider_rope

BASELINE = "a6f14b520dbc31022ffaa971d111619926031b17"
KERNEL = "rl_engine/backends/shared/triton/rope/h3_mm_rope.py"
torch.set_num_threads(1)
torch.manual_seed(656)

def compare(path):
    source = subprocess.check_output(["git", "show", BASELINE + ":" + KERNEL])
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("pr656_before", path)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    operators = {"old": old.TritonH3MMRoPEOp(), "new": TritonH3MMRoPEOp()}
    rows = []
    for layout in ("tiny", "medium"):
        positions = h3_fl2va_layout(**LAYOUTS[layout], device="cuda")["position_ids"]
        tables = provider_rope(positions)
        shape = (1, len(positions), 56, 128)
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn_like(x)
        for strided in (False, True):
            operands = (x, *tables, dy)
            if strided:
                operands = tuple(torch.stack((t, t), dim=-1)[..., 0] for t in operands)
            for mode in ("forward", "backward"):
                calls = {name: timed_call(op, mode, *operands) for name, op in operators.items()}
                assert torch.equal(calls["old"]().view(torch.int16), calls["new"]().view(torch.int16))
                samples = {"old": [], "new": []}
                for iteration in range(180):
                    order = ("old", "new") if iteration % 2 == 0 else ("new", "old")
                    for name in order:
                        torch.cuda.synchronize()
                        start = time.perf_counter_ns()
                        result = calls[name]()
                        elapsed_us = (time.perf_counter_ns() - start) / 1000
                        torch.cuda.synchronize()
                        del result
                        if iteration >= 30:
                            samples[name].append(elapsed_us)
                rows.append(dict(shape=shape, strided=strided, mode=mode, samples_us=samples,
                                 median_us={k: statistics.median(v) for k, v in samples.items()}))
    return rows

with tempfile.TemporaryDirectory() as tmp:
    rows = compare(Path(tmp) / "pr656_before.py")
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
Path(sys.argv[1]).write_text(json.dumps({"baseline": BASELINE, "candidate": head, "dtype": "bf16",
    "method": "public call, idle stream synchronized before and after the CPU timer; warmed and interleaved",
    "warmup": 30, "repeat": 150, "rows": rows}, indent=2) + "\n")
