# MiniMax-H3 partial MM-RoPE on NVIDIA A40

Measurements for [RL-Kernel PR #656](https://github.com/RL-Align/RL-Kernel/pull/656) at [`2c536198`](https://github.com/Jungle430/RL-Kernel/commit/2c536198e9b9e46c2073d1554110524ba5193067), targeting `test-h3` at `60c6aaf9`. The GPU run used the same Git tree as this commit. The final real-weight replay also records its clean checkout in the JSON.

| Check | Result |
| --- | --- |
| Standalone RoPE, registry, checker inputs and benchmark tests, two A40s | 144 passed |
| CUDA unified checker, Native and Triton, BF16 and FP32, forward and input gradient | 4/4 passed |
| Legacy scalar-stride path, PyTorch 2.6.0+cu124 and Triton 3.2.0 | 26 selected tests passed |
| Pinned MiniMax-H3 block-0 weights, model/review/report tests | 40 passed, 2 failed at `norm1`, 1 SM90-only test skipped |
| Unmodified target branch, the same two failing forward cases | Both failed at `norm1` |
| Swapping only candidate RoPE for the provider in block-0 | All 17 forward-node hashes matched in each of tiny and small. Both RoPE nodes passed isolated acceptance in both layouts. |

[Check outputs](checks/) · [Model-boundary block replay](block-replay-final.json) · [Controlled RoPE substitution](block-ab.json) · [A/B replay script](block_ab.py)

## Benchmark

A single A40 ran all 16 combinations of `[1, S, H, 128]`, where `S` is 264, 925, 3,160 or 73,426 and `H` is 14 or 56, in BF16/FP32. Each combination passed acceptance before forward, backward and forward+backward timings began. Each provider used 10 warmups and 40 interleaved CUDA-event samples in eager mode with one CPU thread. Backward reused a prebuilt graph, and the combined path built a fresh graph per call. Speedup divides the baseline median by the Triton median. The original eager H3 BF16 backward rounds products earlier than Native and Triton.

[All medians, memory and diagnostics](benchmark/report.md) · [Raw samples](benchmark/results.json) · [Latency](figures/h3-mm-rope-latency.png) · [Speedup](figures/h3-mm-rope-speedup.png) · [Peak allocation](figures/h3-mm-rope-memory.png)

An additional paired test compares hot CPU submission for the PR's previous and current Triton kernels. Each public call used an idle CUDA stream and 150 timed samples after 30 warmups. Synchronization before and after the call stayed outside the timer. This measures host-side submission with warmed allocations and kernels. [Measurements](submit.json) · [Reproduction script](compare_submit.py)

## Reproduce

From the tested source checkout on NVIDIA CUDA, with its H3 extensions built, run:

```bash
python -m pytest -q tests/ops/rope/test_h3_mm_rope.py tests/benchmarks/test_h3_mm_rope_benchmark.py tests/validation/operators/test_operator_inputs.py tests/runtime/test_kernel_registry.py
python tools/weights/prepare_h3_weights.py --out /tmp/h3-weights --block
export RL_KERNEL_H3_WEIGHTS=/tmp/h3-weights
python -m pytest -q tests/models/minimax_h3/test_h3_one_block.py tests/models/minimax_h3/test_h3_review_fixes.py tests/models/minimax_h3/test_h3_report.py
python tools/validation/models/h3_block_replay.py --layouts tiny,small --timing-repeats 0 --out /tmp/block-replay.json
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.models.benchmark_h3_mm_rope --seed 656 --warmup 10 --repeat 40 --output-dir /tmp/h3-rope
python tools/benchmarking/plot_h3_mm_rope.py /tmp/h3-rope/results.json --out /tmp/h3-rope/figures
```

[Source, environment and weight identity](provenance.json) · [File checksums](SHA256SUMS)
