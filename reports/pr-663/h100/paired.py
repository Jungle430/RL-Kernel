import importlib.util
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

out = Path("/workspace/pr663-result")
cpu_list = Path("/sys/devices/system/node/node1/cpulist").read_text().strip()
near = set()
for part in cpu_list.split(","):
    group = [int(v) for v in part.split("-")]
    near.update(range(group[0], group[-1] + 1))
cpu = min(near & os.sched_getaffinity(0))
os.sched_setaffinity(0, {cpu})
import torch
from benchmarks.operators.embedding.benchmark_index_select import make_inputs
from rl_engine.backends.shared.triton.embedding.index_select import TritonIndexSelectOp

prior = Path("/workspace/pr663-prior")
spec = importlib.util.spec_from_file_location("pr663_previous", prior / "rl_engine/backends/shared/triton/embedding/index_select.py")
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
old_op, new_op = previous.TritonIndexSelectOp(), TritonIndexSelectOp()
torch.set_num_threads(1)
ops = {"torch": lambda x,i: x.index_select(1,i), "current": lambda x,i: new_op(x,i,dim=1,unique_indices=True), "previous": lambda x,i: old_op(x,i,dim=1,unique_indices=True)}
orders = list(itertools.permutations(ops))
def commit(path):
    return subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()

report = {"current":commit("/workspace/pr663-src"), "previous":commit(prior), "gpu":torch.cuda.get_device_name(0), "cpu_affinity":cpu, "warmup":12, "samples":120, "seed":420, "method":"Eager, GPU 0. Same inputs, graph convention and synchronized CUDA-event wall-time boundaries as the PR benchmark. All six backend orders rotate equally within each workload. One GPU-local CPU core, one PyTorch CPU thread.", "results":[]}
begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
for width, size in ((32,129),(96,129),(32,38734),(96,38734)):
    for contiguous in (True, False):
        shape = (1, size, width)
        x, index, grad = make_inputs(shape, 1, torch.float32, "cuda", 420, x_contiguous=contiguous, grad_y_contiguous=contiguous)
        for phase in ("forward", "backward", "forward_backward"):
            calls = {}
            def call(op, leaf, y, phase):
                if phase == "forward":
                    return op(leaf,index)
                if phase == "backward":
                    return torch.autograd.grad(y,leaf,grad,retain_graph=True)[0]
                return torch.autograd.grad(op(leaf,index),leaf,grad)[0]
            for name, op in ops.items():
                leaf=x.detach().requires_grad_()
                y=op(leaf,index)
                calls[name]=lambda op=op,leaf=leaf,y=y:call(op,leaf,y,phase)
            for j in range(12):
                for name in orders[j % 6]:
                    calls[name]()
            torch.cuda.synchronize()
            data={k:[] for k in ops}
            for j in range(120):
                for name in orders[j % 6]:
                    start=time.perf_counter_ns()
                    begin.record()
                    calls[name]()
                    end.record()
                    end.synchronize()
                    data[name].append((time.perf_counter_ns()-start)/1000)
            report["results"].append({"shape":shape,"layout":"C" if contiguous else "S","phase":phase,"wall_us":data})
(out/"paired.json").write_text(json.dumps(report))
