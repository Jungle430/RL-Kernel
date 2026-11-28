"""Validate every output and input-gradient element of a channel-major H3 view."""
import json
import subprocess
import sys
from pathlib import Path

import torch
import triton

from rl_engine.backends.shared.triton.rope.h3_mm_rope import TritonH3MMRoPEOp
from rl_engine.reference.rope.h3_mm_rope import NativeH3MMRoPEOp


torch.set_num_threads(1)
B, S, H, D = 8, 73426, 56, 128
dtype = torch.bfloat16
row = torch.arange(1, D + 1, dtype=dtype, device="cuda")
base = torch.empty((D, B, S, H), device="cuda", dtype=dtype)
base.copy_(row[:, None, None, None])
x = base.permute(1, 2, 3, 0).detach().requires_grad_()
assert x.shape == (B, S, H, D) and x.stride(-1) == B*S*H
cos = torch.full((S, 96), 0.5, dtype=torch.float32, device="cuda")
sin = torch.full((S, 96), -0.25, dtype=torch.float32, device="cuda")

gold_x = row.view(1, 1, 1, D).detach().requires_grad_()
gold_y = NativeH3MMRoPEOp()(gold_x, cos[:1], sin[:1])
gold_dx, = torch.autograd.grad(gold_y, gold_x, grad_outputs=gold_x.detach())
y = TritonH3MMRoPEOp()(x, cos, sin)
dx, = torch.autograd.grad(y, x, grad_outputs=x.detach())
def matches(tensor, gold):
    expected = gold.detach().view(torch.int16).reshape(1, 1, D)
    for batch in tensor.detach().view(torch.int16):
        for chunk in batch.split(2048, dim=0):
            if not torch.equal(chunk, expected.expand_as(chunk)):
                return False
    return True

result = dict(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              gpu=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
              dtype="bfloat16", shape=list(x.shape), strides=list(x.stride()),
              channel_96_element_offset=96*x.stride(-1), elements_per_direction=x.numel(),
              forward_bitwise=matches(y, gold_y), input_gradient_bitwise=matches(dx, gold_dx))
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
assert result["forward_bitwise"] and result["input_gradient_bitwise"]
