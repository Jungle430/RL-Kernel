# MiniMax-H3 partial RoPE regression on NVIDIA A40

[RL-Kernel PR #656](https://github.com/RL-Align/RL-Kernel/pull/656) was tested at [`57c8127c`](https://github.com/Jungle430/RL-Kernel/commit/57c8127c8f9a04940360c5d857bfe470c2b209b2) against `test-h3` at `60c6aaf9`. The replay records the clean source tree and the pinned MiniMax-H3 weight revision `42ed227ee7df40d41602854ae760620d6eb651fe`. [Run identity and test counts](provenance.json)

| Suite | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Operator, registry, benchmark and checker on two A40s | 152 | 0 | 0 |
| Benchmark tests under python -O | 14 | 0 | 0 |
| PyTorch 2.6 and Triton 3.2 | 27 | 0 | 0 |
| H3 block zero, review regressions and reports | 44 | 2 | 1 |
| Unmodified target branch, matching block-zero cases | 0 | 2 | 0 |
| H3 text refiner on this revision | 12 | 4 | 0 |
| Unmodified target, same text-refiner suite | 12 | 4 | 0 |
| Previous PR revision 61492458, same text-refiner suite | 12 | 4 | 0 |
| H3 operator traces, including one real block | 32 | 0 | 0 |
| Local CPU suites | 181 | 0 | 66 |

Both `tiny` and `small` fail the `norm1` bitwise comparison on the candidate and the unmodified target. Isolated Q and K RoPE pass in both layouts. Substituting original H3 RoPE preserves all 17 forward-node hashes per layout. A separate tiny replay with one deliberately injected K NaN reports Q passing, K failing with one violation and a JSON null maximum. [Test logs](checks/) · [Block replay](block-replay-final.json) · [RoPE substitution](block-ab.json) · [Injected failure](real-injection.json)

The refiner has no RoPE node. All three revisions fail the same four isolated provider-bitwise checks on A40. Block 0 at lengths 64 and 257 first fails at `norm1`. Block 1 first fails at `q_norm` for length 64 and at `norm1` for length 257. The 12 other cases pass on all three revisions. The simulated ROCm registry test exercises Native selection and reporting. The physical GPU used here was NVIDIA. [Current](checks/refiner.log) · [Target](checks/base-refiner.log) · [Previous PR revision](checks/prev-refiner.log)

## Benchmark

One A40 measured the 16 combinations of packed sequence length, head count and activation dtype in the operator benchmark. Each case passed model-boundary forward and input-gradient acceptance. Each of the 48 timing rows contains 40 interleaved CUDA-event samples per provider after 10 warmups. Original eager H3, Native and Triton use the same inputs. Seed 656, one CPU thread. [Timing table](benchmark/report.md) · [Raw measurements, tolerances and examples](benchmark/results.json) · [Figures](figures/)

Unrounded FP32 comparisons counted 2 candidate forward, 0 candidate input-gradient and 2 original eager input-gradient values outside tolerance. All 16 numeric diagnostic records match the preceding run. The independent model-boundary reference determines acceptance.

## Reproduce

From the tested checkout with the H3 extension built on CUDA, run the commands below. The device-guard cases require two GPUs. The text-refiner CUDA tests read transformer shards 1 and 14 from the same pinned model revision.

```bash
python -m pytest -q tests/ops/rope/test_h3_mm_rope.py tests/benchmarks/test_h3_mm_rope_benchmark.py tests/validation/operators/test_operator_inputs.py tests/runtime/test_kernel_registry.py
python -O -m pytest -q tests/benchmarks/test_h3_mm_rope_benchmark.py
python tools/weights/prepare_h3_weights.py --block --out /tmp/h3-weights
export RL_KERNEL_H3_WEIGHTS=/tmp/h3-weights
hf download MiniMaxAI/MiniMax-H3 transformer/diffusion_pytorch_model-00014-of-00014.safetensors --revision 42ed227ee7df40d41602854ae760620d6eb651fe --local-dir /tmp/h3-weights/hf
python -m pytest -q tests/models/minimax_h3/test_h3_one_block.py tests/models/minimax_h3/test_h3_review_fixes.py tests/models/minimax_h3/test_h3_report.py tests/models/minimax_h3/test_h3_token_refiner.py tests/models/minimax_h3/test_h3_operator_trace.py
python tools/validation/models/h3_block_replay.py --layouts tiny,small --timing-repeats 0 --out /tmp/h3-block-replay.json
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.models.benchmark_h3_mm_rope --seed 656 --warmup 10 --repeat 40 --output-dir /tmp/h3-rope
```

The [previous evidence and measurement scripts](https://github.com/Jungle430/RL-Kernel/tree/7bd5fb7dab3298d5e01e4312c5017b33cae28c5d) document the procedures for the [64-bit stride check](wide-model.json), [paired host timings](submit.json), and peak validation memory. The current validation peak above preallocated inputs was 2,348.8 MiB in BF16 and 4,362.0 MiB in FP32 at `[1,73426,56,128]`. The same allocations before chunking were 10,964.9 MiB in BF16 and 12,707.9 MiB in FP32. [Measurements](checks/).

SHA-256 checksums are listed in [SHA256SUMS](SHA256SUMS).
