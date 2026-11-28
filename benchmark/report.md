# H3 partial MM-RoPE benchmark

Eager public wrappers; event medians; no torch.compile or CUDA Graphs. Backward reuses a graph; forward+backward creates a fresh graph. Inputs, table generation, compilation and checks are outside timing.

Provider = original model expression/autograd. Native = matched forward with explicit FP32 backward. BF16 provider backward rounds products earlier, so its gradient is not required to be bitwise equal.

Environment: `{'python': '3.12.3', 'torch': '2.8.0+cu128', 'triton': '3.4.0', 'cuda': '12.8', 'hip': None, 'gpu': 'NVIDIA A40', 'compute_capability': (8, 6), 'cpu_threads': 1}`

Complete: True

Acceptance: model coefficient/product storage boundaries, independent FP32 VJP. Unrounded FP32 mathematical error is diagnostic only; its outlier counts below are not waived or interpreted as a pure FP32 accuracy pass.

| Dtype / shape | Mode | Provider us | Native us | Triton us | Provider / Triton | Native / Triton |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bf16 1x264x14x128 | forward | 157.02 | 279.26 | 111.44 | 1.41x | 2.51x |
| bf16 1x264x14x128 | backward | 511.31 | 672.37 | 445.95 | 1.15x | 1.51x |
| bf16 1x264x14x128 | forward_backward | 704.24 | 961.15 | 456.91 | 1.54x | 2.10x |
| fp32 1x264x14x128 | forward | 173.14 | 221.28 | 141.98 | 1.22x | 1.56x |
| fp32 1x264x14x128 | backward | 367.46 | 357.82 | 239.44 | 1.53x | 1.49x |
| fp32 1x264x14x128 | forward_backward | 650.54 | 652.85 | 492.66 | 1.32x | 1.33x |
| bf16 1x264x56x128 | forward | 189.20 | 357.47 | 126.30 | 1.50x | 2.83x |
| bf16 1x264x56x128 | backward | 289.41 | 347.06 | 194.85 | 1.49x | 1.78x |
| bf16 1x264x56x128 | forward_backward | 700.50 | 980.72 | 438.30 | 1.60x | 2.24x |
| fp32 1x264x56x128 | forward | 204.77 | 230.46 | 112.51 | 1.82x | 2.05x |
| fp32 1x264x56x128 | backward | 530.16 | 525.04 | 317.42 | 1.67x | 1.65x |
| fp32 1x264x56x128 | forward_backward | 594.62 | 653.02 | 421.98 | 1.41x | 1.55x |
| bf16 1x925x14x128 | forward | 158.05 | 308.24 | 112.16 | 1.41x | 2.75x |
| bf16 1x925x14x128 | backward | 431.20 | 569.66 | 295.42 | 1.46x | 1.93x |
| bf16 1x925x14x128 | forward_backward | 675.76 | 932.32 | 489.30 | 1.38x | 1.91x |
| fp32 1x925x14x128 | forward | 189.54 | 216.34 | 116.00 | 1.63x | 1.86x |
| fp32 1x925x14x128 | backward | 268.37 | 241.82 | 157.57 | 1.70x | 1.53x |
| fp32 1x925x14x128 | forward_backward | 552.02 | 568.27 | 376.13 | 1.47x | 1.51x |
| bf16 1x925x56x128 | forward | 356.91 | 891.92 | 141.73 | 2.52x | 6.29x |
| bf16 1x925x56x128 | backward | 561.50 | 771.33 | 326.82 | 1.72x | 2.36x |
| bf16 1x925x56x128 | forward_backward | 839.84 | 1500.38 | 496.48 | 1.69x | 3.02x |
| fp32 1x925x56x128 | forward | 542.99 | 566.11 | 179.55 | 3.02x | 3.15x |
| fp32 1x925x56x128 | backward | 1068.51 | 902.70 | 719.42 | 1.49x | 1.25x |
| fp32 1x925x56x128 | forward_backward | 1339.79 | 1086.18 | 507.66 | 2.64x | 2.14x |
| bf16 1x3160x14x128 | forward | 348.77 | 820.72 | 171.41 | 2.03x | 4.79x |
| bf16 1x3160x14x128 | backward | 469.26 | 678.96 | 260.80 | 1.80x | 2.60x |
| bf16 1x3160x14x128 | forward_backward | 737.97 | 1311.58 | 478.67 | 1.54x | 2.74x |
| fp32 1x3160x14x128 | forward | 497.04 | 528.16 | 246.75 | 2.01x | 2.14x |
| fp32 1x3160x14x128 | backward | 781.20 | 567.01 | 338.24 | 2.31x | 1.68x |
| fp32 1x3160x14x128 | forward_backward | 1158.30 | 946.82 | 450.16 | 2.57x | 2.10x |
| bf16 1x3160x56x128 | forward | 1030.46 | 2754.46 | 372.48 | 2.77x | 7.39x |
| bf16 1x3160x56x128 | backward | 1431.54 | 2003.97 | 505.68 | 2.83x | 3.96x |
| bf16 1x3160x56x128 | forward_backward | 2354.70 | 4556.66 | 692.62 | 3.40x | 6.58x |
| fp32 1x3160x56x128 | forward | 1691.26 | 1720.21 | 610.77 | 2.77x | 2.82x |
| fp32 1x3160x56x128 | backward | 2552.06 | 1687.01 | 698.59 | 3.65x | 2.41x |
| fp32 1x3160x56x128 | forward_backward | 4135.36 | 3258.11 | 1121.52 | 3.69x | 2.91x |
| bf16 1x73426x14x128 | forward | 5663.23 | 15558.83 | 2442.40 | 2.32x | 6.37x |
| bf16 1x73426x14x128 | backward | 7740.35 | 10854.91 | 2548.34 | 3.04x | 4.26x |
| bf16 1x73426x14x128 | forward_backward | 13242.70 | 26294.70 | 4762.75 | 2.78x | 5.52x |
| fp32 1x73426x14x128 | forward | 9641.28 | 9669.90 | 3452.27 | 2.79x | 2.80x |
| fp32 1x73426x14x128 | backward | 14293.38 | 9276.66 | 3622.14 | 3.95x | 2.56x |
| fp32 1x73426x14x128 | forward_backward | 23771.09 | 18754.64 | 6834.13 | 3.48x | 2.74x |
| bf16 1x73426x56x128 | forward | 22825.25 | 61196.32 | 9417.31 | 2.42x | 6.50x |
| bf16 1x73426x56x128 | backward | 30969.09 | 42265.01 | 9649.12 | 3.21x | 4.38x |
| bf16 1x73426x56x128 | forward_backward | 53513.04 | 103219.83 | 18670.13 | 2.87x | 5.53x |
| fp32 1x73426x56x128 | forward | 38289.28 | 38311.28 | 13487.73 | 2.84x | 2.84x |
| fp32 1x73426x56x128 | backward | 56565.14 | 36498.56 | 13735.71 | 4.12x | 2.66x |
| fp32 1x73426x56x128 | forward_backward | 94552.86 | 74450.32 | 26837.01 | 3.52x | 2.77x |

## Unrounded FP32 mathematical diagnostics (not an acceptance gate)

| Dtype / shape | Direction | Outside tolerance / elements | Max abs error | Max error / limit |
| --- | --- | ---: | ---: | ---: |
| bf16 1x264x14x128 | forward | 0 / 473088 | 0.031326294 | 0.56876 |
| bf16 1x264x14x128 | backward | 0 / 473088 | 0.021414757 | 0.3712 |
| bf16 1x264x14x128 | provider_backward | 0 / 473088 | 0.021697044 | 0.56328 |
| fp32 1x264x14x128 | forward | 0 / 473088 | 0 | 0 |
| fp32 1x264x14x128 | backward | 0 / 473088 | 0 | 0 |
| fp32 1x264x14x128 | provider_backward | 0 / 473088 | 0 | 0 |
| bf16 1x264x56x128 | forward | 0 / 1892352 | 0.029600143 | 0.58183 |
| bf16 1x264x56x128 | backward | 0 / 1892352 | 0.01874876 | 0.37124 |
| bf16 1x264x56x128 | provider_backward | 0 / 1892352 | 0.032126904 | 0.60464 |
| fp32 1x264x56x128 | forward | 0 / 1892352 | 0 | 0 |
| fp32 1x264x56x128 | backward | 0 / 1892352 | 0 | 0 |
| fp32 1x264x56x128 | provider_backward | 0 / 1892352 | 0 | 0 |
| bf16 1x925x14x128 | forward | 0 / 1657600 | 0.032326698 | 0.74429 |
| bf16 1x925x14x128 | backward | 0 / 1657600 | 0.019310474 | 0.36997 |
| bf16 1x925x14x128 | provider_backward | 0 / 1657600 | 0.027178288 | 0.62934 |
| fp32 1x925x14x128 | forward | 0 / 1657600 | 0 | 0 |
| fp32 1x925x14x128 | backward | 0 / 1657600 | 0 | 0 |
| fp32 1x925x14x128 | provider_backward | 0 / 1657600 | 0 | 0 |
| bf16 1x925x56x128 | forward | 0 / 6630400 | 0.033035278 | 0.89688 |
| bf16 1x925x56x128 | backward | 0 / 6630400 | 0.02532959 | 0.52654 |
| bf16 1x925x56x128 | provider_backward | 0 / 6630400 | 0.037958145 | 0.89208 |
| fp32 1x925x56x128 | forward | 0 / 6630400 | 0 | 0 |
| fp32 1x925x56x128 | backward | 0 / 6630400 | 0 | 0 |
| fp32 1x925x56x128 | provider_backward | 0 / 6630400 | 0 | 0 |
| bf16 1x3160x14x128 | forward | 0 / 5662720 | 0.034149647 | 0.65835 |
| bf16 1x3160x14x128 | backward | 0 / 5662720 | 0.024774551 | 0.45088 |
| bf16 1x3160x14x128 | provider_backward | 0 / 5662720 | 0.034576416 | 0.72684 |
| fp32 1x3160x14x128 | forward | 0 / 5662720 | 0 | 0 |
| fp32 1x3160x14x128 | backward | 0 / 5662720 | 0 | 0 |
| fp32 1x3160x14x128 | provider_backward | 0 / 5662720 | 0 | 0 |
| bf16 1x3160x56x128 | forward | 0 / 22650880 | 0.03349638 | 0.69973 |
| bf16 1x3160x56x128 | backward | 0 / 22650880 | 0.023727417 | 0.4601 |
| bf16 1x3160x56x128 | provider_backward | 0 / 22650880 | 0.040925026 | 0.81032 |
| fp32 1x3160x56x128 | forward | 0 / 22650880 | 0 | 0 |
| fp32 1x3160x56x128 | backward | 0 / 22650880 | 0 | 0 |
| fp32 1x3160x56x128 | provider_backward | 0 / 22650880 | 0 | 0 |
| bf16 1x73426x14x128 | forward | 2 / 131579392 | 0.03861475 | 1.0864 |
| bf16 1x73426x14x128 | backward | 0 / 131579392 | 0.025792599 | 0.52215 |
| bf16 1x73426x14x128 | provider_backward | 0 / 131579392 | 0.036728382 | 0.92508 |
| fp32 1x73426x14x128 | forward | 0 / 131579392 | 0 | 0 |
| fp32 1x73426x14x128 | backward | 0 / 131579392 | 0 | 0 |
| fp32 1x73426x14x128 | provider_backward | 0 / 131579392 | 0 | 0 |
| bf16 1x73426x56x128 | forward | 0 / 526317568 | 0.03937006 | 0.95413 |
| bf16 1x73426x56x128 | backward | 0 / 526317568 | 0.025775433 | 0.52519 |
| bf16 1x73426x56x128 | provider_backward | 2 / 526317568 | 0.038446426 | 1.1438 |
| fp32 1x73426x56x128 | forward | 0 / 526317568 | 0 | 0 |
| fp32 1x73426x56x128 | backward | 0 / 526317568 | 0 | 0 |
| fp32 1x73426x56x128 | provider_backward | 0 / 526317568 | 0 | 0 |
