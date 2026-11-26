# CUDA regression for shared index selection on MiniMax-H3

Commit `9f98c250666828ac0b9faaf5797671f98e2775d6` ran on two NVIDIA H100 SXM 80 GB GPUs (SM90). The test image was `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` with Python 3.12.3, PyTorch 2.8.0+cu128, Triton 3.4.0, CUDA 12.8 and driver 595.112.07.

| Check | Result |
| --- | --- |
| Operator, benchmark definitions, dispatch, registry and validation inputs | 243 passed, including 4 noncurrent-device cases and 16 debug/release bounds cases |
| Released Diffusers 0.43.0, H3 packed forward and full-model binding contracts | 55 passed |
| Pinned Diffusers `f53d552036a0d1bd5570782a39cd40cfabf112bc`, H3 packed forward | 19 passed |
| Triton checker with random upstream gradients | Six runs. Forward and input-gradient maximum absolute error was 0 for all three dtypes and both projection widths |
| H3 example | Registry selected `TritonIndexSelectOp`. Video shape was `(2, 5, 96)`, audio was `(2, 4, 32)` |

All three pytest runs completed with zero failures, errors or skips. The H3 tests exercise the provider forward on a small fixture. They compare video and audio outputs and the fixture's input gradients byte for byte with the provider, run the compiled CUDA packing extension, and exercise the case where output selection uses Triton on its own.

## Reproduction

From the tested RL-Kernel checkout, with CUDA PyTorch and its matching development toolkit installed, run the following commands. The first model invocation uses PyPI Diffusers 0.43.0. For the last one, set `DIFFUSERS_REPO` to a separate checkout of the pinned Diffusers revision.

```bash
python -m pip install ninja pytest pyyaml tabulate matplotlib accelerate 'diffusers==0.43.0' 'transformers==5.13.1'
MAX_JOBS=8 RL_KERNEL_REQUIRE_EXT=1 TORCH_CUDA_ARCH_LIST=9.0 python setup.py build_ext --inplace
export PYTHONPATH=.
python -m pytest -q tests/ops/embedding/test_index_select.py tests/benchmarks/test_index_select_benchmark.py tests/runtime/test_dispatch.py tests/runtime/test_kernel_registry.py tests/validation/operators/test_operator_inputs.py
python -m pytest -q tests/models/minimax_h3/test_h3_packed_forward.py tests/models/minimax_h3/test_h3_full_model.py
PYTHONPATH="$DIFFUSERS_REPO/src:." python -m pytest -q tests/models/minimax_h3/test_h3_packed_forward.py
python examples/models/minimax_h3/packed_index_select.py
```

For the operator checker, use `--select-shape 2 17 96 --select-dim -2` for video and `--select-shape 2 17 32 --select-dim 1` for audio. Run each case with `--dtype fp32`, `fp16` and `bf16`.

```bash
python tools/validation/operators/check_operator.py --op index_select --candidate triton --device cuda --dtype fp32 --select-shape 2 17 96 --select-dim -2 --check-grad --grad-mode random --json
```

To reproduce the layout benchmark, set `GPU_LOCAL_CPU` to a CPU in GPU 0's NUMA node. This host used core 56. The benchmark ran on a single GPU with one PyTorch CPU thread, without CUDA Graph replay. Each timing includes host dispatch and device synchronization.

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
taskset -c "$GPU_LOCAL_CPU" python benchmarks/operators/embedding/benchmark_index_select.py --shapes 1x129x32 1x129x96 1x38734x32 1x38734x96 --dim 1 --dtype float32 --warmup 10 --samples 100 --seed 420 --output reports/index-select/h3.json
MPLBACKEND=Agg python benchmarks/operators/embedding/report_index_select.py reports/index-select/h3.json --output-dir reports/index-select
```

[Pytest case list and six numerical reports](validation.json) · [Benchmarks and figures](summary.md) · [Raw benchmark samples](h3.json) · [Comparison with the previous launcher](launch-comparison.md)
