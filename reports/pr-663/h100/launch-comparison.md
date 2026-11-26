# Same-process comparison with the previous Triton launcher

GPU 0 was an H100 SXM 80 GB. The process used one GPU-local CPU core and one PyTorch CPU thread. All three implementations used identical inputs and upstream gradients. Each phase ran 12 warmups followed by 120 synchronized wall-time samples per implementation. The six execution orders occurred equally often. The index was contiguous and selected half of the sequence without replacement.

The current commit is `9f98c250666828ac0b9faaf5797671f98e2775d6` and the previous launcher is `6d1b460737f7d650e525f4c5abb8a5b1fc3a3168`. The FP32 cases below show median times in µs. C means contiguous and S means a stride-two view. Each row uses the same layout for both tensors. The retained-graph backward differentiates a forward built with the indicated layout.

| Input shape | Layout | Phase | PyTorch | Current Triton | Previous Triton | Current time reduction vs previous |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 × 129 × 32 | C | Forward | 35.1 | 72.8 | 94.6 | 23.1% |
| 1 × 129 × 32 | C | Backward | 70.3 | 111.0 | 117.4 | 5.4% |
| 1 × 129 × 32 | C | Forward + backward | 99.1 | 183.7 | 221.7 | 17.1% |
| 1 × 129 × 32 | S | Forward | 35.4 | 78.4 | 99.9 | 21.5% |
| 1 × 129 × 32 | S | Backward | 71.2 | 116.2 | 123.5 | 5.9% |
| 1 × 129 × 32 | S | Forward + backward | 100.0 | 198.9 | 237.6 | 16.3% |
| 1 × 129 × 96 | C | Forward | 35.1 | 72.9 | 94.2 | 22.6% |
| 1 × 129 × 96 | C | Backward | 70.5 | 109.8 | 116.8 | 5.9% |
| 1 × 129 × 96 | C | Forward + backward | 98.6 | 187.1 | 223.1 | 16.1% |
| 1 × 129 × 96 | S | Forward | 35.4 | 78.9 | 100.2 | 21.2% |
| 1 × 129 × 96 | S | Backward | 71.2 | 116.7 | 123.2 | 5.3% |
| 1 × 129 × 96 | S | Forward + backward | 101.9 | 205.4 | 245.4 | 16.3% |
| 1 × 38734 × 32 | C | Forward | 39.9 | 73.3 | 94.7 | 22.6% |
| 1 × 38734 × 32 | C | Backward | 70.1 | 109.5 | 116.8 | 6.2% |
| 1 × 38734 × 32 | C | Forward + backward | 100.6 | 188.7 | 228.2 | 17.3% |
| 1 × 38734 × 32 | S | Forward | 35.4 | 78.7 | 100.1 | 21.4% |
| 1 × 38734 × 32 | S | Backward | 71.6 | 116.6 | 123.5 | 5.6% |
| 1 × 38734 × 32 | S | Forward + backward | 102.8 | 204.8 | 245.6 | 16.6% |
| 1 × 38734 × 96 | C | Forward | 39.9 | 72.4 | 94.2 | 23.1% |
| 1 × 38734 × 96 | C | Backward | 70.9 | 111.0 | 118.1 | 6.0% |
| 1 × 38734 × 96 | C | Forward + backward | 101.1 | 190.3 | 228.8 | 16.8% |
| 1 × 38734 × 96 | S | Forward | 36.3 | 78.4 | 100.2 | 21.8% |
| 1 × 38734 × 96 | S | Backward | 71.2 | 116.1 | 123.2 | 5.8% |
| 1 × 38734 × 96 | S | Forward + backward | 103.7 | 207.3 | 250.4 | 17.2% |

[Individual samples](paired.json) · [Harness](paired.py)
