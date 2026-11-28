"""Measure CUDA allocations attributable to check_case with identical inputs on parent and head."""
import argparse
import gc
import json
import subprocess
from pathlib import Path

import torch
from benchmarks.models import benchmark_h3_mm_rope as b
from rl_engine.backends.shared.triton.rope.h3_mm_rope import TritonH3MMRoPEOp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("revision", choices=("parent", "head"))
parser.add_argument("dtype", choices=("bf16","fp32"))
parser.add_argument("output", type=Path)
args = parser.parse_args()
commit = subprocess.check_output(["git", "rev-parse", "HEAD^" if args.revision == "parent" else "HEAD"], text=True).strip()
module_path = "benchmarks/models/benchmark_h3_mm_rope.py"
namespace = {"__name__": "measured_check_case"}
source = subprocess.check_output(["git", "show", f"{commit}:{module_path}"], text=True)
exec(compile(source, module_path, "exec"), namespace)
check = namespace["check_case"]
torch.set_num_threads(1)
dtype = b.DTYPES[args.dtype]
positions = b.h3_fl2va_layout(**b.LAYOUTS["long"], device="cuda")["position_ids"]
cos, sin = b.provider_rope(positions)
shape = (1,len(positions),56,128)
gen = torch.Generator(device="cuda").manual_seed(656)
x = torch.randn(shape,device="cuda",dtype=dtype,generator=gen)
upstream = torch.randn(shape,device="cuda",dtype=dtype,generator=gen)
operators = {"native":b.NativeH3MMRoPEOp(), "triton":TritonH3MMRoPEOp()}
# Initialize both libraries and kernels before measuring allocator high water.
check(operators, x[:,:8], cos[:8], sin[:8], upstream[:,:8])
gc.collect()
torch.cuda.synchronize()
torch.cuda.empty_cache()
baseline = torch.cuda.memory_allocated()
torch.cuda.reset_peak_memory_stats()
checks = check(operators,x,cos,sin,upstream)
torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated()
info = dict(revision=args.revision, commit=commit, shape=shape, dtype=args.dtype,
            gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), seed=656,
            baseline_allocated_bytes=baseline, peak_allocated_bytes=peak,
            validation_extra_peak_bytes=peak-baseline, checks=checks)
args.output.write_text(json.dumps(info,indent=2)+"\n")
print(args.revision,args.dtype, "baseline MiB",baseline/2**20, "incremental validation peak MiB",(peak-baseline)/2**20)
