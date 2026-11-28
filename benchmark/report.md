# H3 partial MM-RoPE benchmark

Eager public wrappers; event medians; no torch.compile or CUDA Graphs. Backward reuses a graph; forward+backward creates a fresh graph. Inputs, table generation, compilation and checks are outside timing.

Provider = original model expression/autograd. Native = matched forward with explicit FP32 backward. BF16 provider backward rounds products earlier, so its gradient is not required to be bitwise equal.

Environment: `{'python': '3.12.3', 'torch': '2.8.0+cu128', 'triton': '3.4.0', 'cuda': '12.8', 'hip': None, 'gpu': 'NVIDIA A40', 'compute_capability': (8, 6), 'cpu_threads': 1}`

Complete: True

Acceptance: model coefficient/product storage boundaries, independent FP32 VJP. Unrounded FP32 mathematical error is diagnostic only; its outlier counts below are not waived or interpreted as a pure FP32 accuracy pass.

| Dtype / shape | Mode | Provider us | Native us | Triton us | Provider / Triton | Native / Triton |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bf16 1x264x14x128 | forward | 159.14 | 282.56 | 113.46 | 1.40x | 2.49x |
| bf16 1x264x14x128 | backward | 801.52 | 872.42 | 513.84 | 1.56x | 1.70x |
| bf16 1x264x14x128 | forward_backward | 1152.22 | 1603.36 | 989.12 | 1.16x | 1.62x |
| fp32 1x264x14x128 | forward | 151.31 | 197.84 | 124.78 | 1.21x | 1.59x |
| fp32 1x264x14x128 | backward | 496.18 | 477.95 | 402.03 | 1.23x | 1.19x |
| fp32 1x264x14x128 | forward_backward | 1154.10 | 1111.25 | 834.66 | 1.38x | 1.33x |
| bf16 1x264x56x128 | forward | 168.62 | 352.16 | 127.17 | 1.33x | 2.77x |
| bf16 1x264x56x128 | backward | 974.45 | 1216.75 | 743.26 | 1.31x | 1.64x |
| bf16 1x264x56x128 | forward_backward | 1607.90 | 2124.08 | 1134.61 | 1.42x | 1.87x |
| fp32 1x264x56x128 | forward | 244.27 | 290.66 | 177.09 | 1.38x | 1.64x |
| fp32 1x264x56x128 | backward | 1140.51 | 1097.90 | 814.18 | 1.40x | 1.35x |
| fp32 1x264x56x128 | forward_backward | 1119.95 | 1089.28 | 759.71 | 1.47x | 1.43x |
| bf16 1x925x14x128 | forward | 171.54 | 315.65 | 116.50 | 1.47x | 2.71x |
| bf16 1x925x14x128 | backward | 337.97 | 431.41 | 228.66 | 1.48x | 1.89x |
| bf16 1x925x14x128 | forward_backward | 969.50 | 1144.93 | 623.68 | 1.55x | 1.84x |
| fp32 1x925x14x128 | forward | 188.50 | 215.97 | 117.97 | 1.60x | 1.83x |
| fp32 1x925x14x128 | backward | 880.11 | 924.46 | 721.09 | 1.22x | 1.28x |
| fp32 1x925x14x128 | forward_backward | 1076.00 | 1095.73 | 779.49 | 1.38x | 1.41x |
| bf16 1x925x56x128 | forward | 362.38 | 901.65 | 157.28 | 2.30x | 5.73x |
| bf16 1x925x56x128 | backward | 639.14 | 952.85 | 494.22 | 1.29x | 1.93x |
| bf16 1x925x56x128 | forward_backward | 1222.96 | 1684.21 | 850.11 | 1.44x | 1.98x |
| fp32 1x925x56x128 | forward | 571.54 | 601.92 | 209.20 | 2.73x | 2.88x |
| fp32 1x925x56x128 | backward | 922.82 | 686.14 | 504.88 | 1.83x | 1.36x |
| fp32 1x925x56x128 | forward_backward | 1322.42 | 1073.62 | 625.52 | 2.11x | 1.72x |
| bf16 1x3160x14x128 | forward | 324.10 | 792.13 | 163.95 | 1.98x | 4.83x |
| bf16 1x3160x14x128 | backward | 549.23 | 783.06 | 405.74 | 1.35x | 1.93x |
| bf16 1x3160x14x128 | forward_backward | 1037.39 | 1425.89 | 799.66 | 1.30x | 1.78x |
| fp32 1x3160x14x128 | forward | 484.94 | 516.66 | 227.76 | 2.13x | 2.27x |
| fp32 1x3160x14x128 | backward | 850.86 | 633.36 | 440.43 | 1.93x | 1.44x |
| fp32 1x3160x14x128 | forward_backward | 1232.38 | 1133.73 | 883.46 | 1.39x | 1.28x |
| bf16 1x3160x56x128 | forward | 1013.42 | 2733.94 | 368.26 | 2.75x | 7.42x |
| bf16 1x3160x56x128 | backward | 1546.99 | 2179.14 | 605.33 | 2.56x | 3.60x |
| bf16 1x3160x56x128 | forward_backward | 2364.56 | 4563.41 | 748.24 | 3.16x | 6.10x |
| fp32 1x3160x56x128 | forward | 1688.66 | 1712.38 | 599.20 | 2.82x | 2.86x |
| fp32 1x3160x56x128 | backward | 2657.63 | 1810.24 | 826.67 | 3.21x | 2.19x |
| fp32 1x3160x56x128 | forward_backward | 4190.22 | 3318.22 | 1227.31 | 3.41x | 2.70x |
| bf16 1x73426x14x128 | forward | 5685.68 | 15600.54 | 2490.64 | 2.28x | 6.26x |
| bf16 1x73426x14x128 | backward | 7921.42 | 11031.17 | 2788.06 | 2.84x | 3.96x |
| bf16 1x73426x14x128 | forward_backward | 13336.93 | 26343.44 | 4856.56 | 2.75x | 5.42x |
| fp32 1x73426x14x128 | forward | 9680.56 | 9708.80 | 3515.98 | 2.75x | 2.76x |
| fp32 1x73426x14x128 | backward | 14563.31 | 9590.06 | 3933.47 | 3.70x | 2.44x |
| fp32 1x73426x14x128 | forward_backward | 23830.38 | 18900.37 | 6923.25 | 3.44x | 2.73x |
| bf16 1x73426x56x128 | forward | 22920.29 | 61300.80 | 9577.78 | 2.39x | 6.40x |
| bf16 1x73426x56x128 | backward | 31241.95 | 42623.84 | 10057.78 | 3.11x | 4.24x |
| bf16 1x73426x56x128 | forward_backward | 53623.57 | 103374.82 | 18805.36 | 2.85x | 5.50x |
| fp32 1x73426x56x128 | forward | 38337.26 | 38400.13 | 13532.11 | 2.83x | 2.84x |
| fp32 1x73426x56x128 | backward | 56626.11 | 36614.69 | 13890.53 | 4.08x | 2.64x |
| fp32 1x73426x56x128 | forward_backward | 94779.17 | 74684.78 | 26996.46 | 3.51x | 2.77x |

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
