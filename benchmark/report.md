# H3 partial MM-RoPE benchmark

Eager public wrappers; event medians; no torch.compile or CUDA Graphs. Backward reuses a graph; forward+backward creates a fresh graph. Inputs, table generation, compilation and checks are outside timing.

Provider = original model expression/autograd. Native = matched forward with explicit FP32 backward. BF16 provider backward rounds products earlier, so its gradient is not required to be bitwise equal.

Environment: `{'python': '3.12.3', 'torch': '2.8.0+cu128', 'triton': '3.4.0', 'cuda': '12.8', 'hip': None, 'gpu': 'NVIDIA A40', 'compute_capability': (8, 6), 'cpu_threads': 1}`

Complete: True

Acceptance: model coefficient/product storage boundaries, independent FP32 VJP. Unrounded FP32 mathematical error is diagnostic only; its outlier counts below are not waived or interpreted as a pure FP32 accuracy pass.

| Dtype / shape | Mode | Provider us | Native us | Triton us | Provider / Triton | Native / Triton |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bf16 1x264x14x128 | forward | 130.10 | 232.82 | 89.38 | 1.46x | 2.60x |
| bf16 1x264x14x128 | backward | 460.61 | 622.50 | 284.38 | 1.62x | 2.19x |
| bf16 1x264x14x128 | forward_backward | 1123.30 | 1543.58 | 678.38 | 1.66x | 2.28x |
| fp32 1x264x14x128 | forward | 135.98 | 173.82 | 108.40 | 1.25x | 1.60x |
| fp32 1x264x14x128 | backward | 652.88 | 651.84 | 403.50 | 1.62x | 1.62x |
| fp32 1x264x14x128 | forward_backward | 1261.17 | 1313.44 | 858.35 | 1.47x | 1.53x |
| bf16 1x264x56x128 | forward | 257.78 | 432.46 | 159.90 | 1.61x | 2.70x |
| bf16 1x264x56x128 | backward | 714.45 | 947.70 | 466.86 | 1.53x | 2.03x |
| bf16 1x264x56x128 | forward_backward | 1407.55 | 2021.65 | 961.95 | 1.46x | 2.10x |
| fp32 1x264x56x128 | forward | 242.67 | 284.69 | 170.94 | 1.42x | 1.67x |
| fp32 1x264x56x128 | backward | 743.23 | 735.65 | 496.67 | 1.50x | 1.48x |
| fp32 1x264x56x128 | forward_backward | 1321.15 | 1377.66 | 904.50 | 1.46x | 1.52x |
| bf16 1x925x14x128 | forward | 240.34 | 424.59 | 157.31 | 1.53x | 2.70x |
| bf16 1x925x14x128 | backward | 745.12 | 971.18 | 481.39 | 1.55x | 2.02x |
| bf16 1x925x14x128 | forward_backward | 1402.93 | 1963.84 | 901.33 | 1.56x | 2.18x |
| fp32 1x925x14x128 | forward | 229.79 | 282.66 | 174.88 | 1.31x | 1.62x |
| fp32 1x925x14x128 | backward | 728.72 | 730.02 | 487.10 | 1.50x | 1.50x |
| fp32 1x925x14x128 | forward_backward | 1283.63 | 1315.17 | 868.62 | 1.48x | 1.51x |
| bf16 1x925x56x128 | forward | 375.49 | 910.59 | 152.69 | 2.46x | 5.96x |
| bf16 1x925x56x128 | backward | 717.15 | 961.74 | 463.31 | 1.55x | 2.08x |
| bf16 1x925x56x128 | forward_backward | 1413.25 | 1976.29 | 891.47 | 1.59x | 2.22x |
| fp32 1x925x56x128 | forward | 556.19 | 579.18 | 188.43 | 2.95x | 3.07x |
| fp32 1x925x56x128 | backward | 933.46 | 733.39 | 454.56 | 2.05x | 1.61x |
| fp32 1x925x56x128 | forward_backward | 1469.38 | 1401.02 | 912.00 | 1.61x | 1.54x |
| bf16 1x3160x14x128 | forward | 369.30 | 841.34 | 189.52 | 1.95x | 4.44x |
| bf16 1x3160x14x128 | backward | 712.70 | 950.42 | 464.80 | 1.53x | 2.04x |
| bf16 1x3160x14x128 | forward_backward | 1390.62 | 1947.79 | 884.26 | 1.57x | 2.20x |
| fp32 1x3160x14x128 | forward | 498.94 | 528.82 | 242.54 | 2.06x | 2.18x |
| fp32 1x3160x14x128 | backward | 850.67 | 717.17 | 465.89 | 1.83x | 1.54x |
| fp32 1x3160x14x128 | forward_backward | 1356.26 | 1312.75 | 863.97 | 1.57x | 1.52x |
| bf16 1x3160x56x128 | forward | 992.74 | 2709.26 | 337.63 | 2.94x | 8.02x |
| bf16 1x3160x56x128 | backward | 1496.35 | 2182.75 | 568.38 | 2.63x | 3.84x |
| bf16 1x3160x56x128 | forward_backward | 2414.50 | 4630.74 | 782.38 | 3.09x | 5.92x |
| fp32 1x3160x56x128 | forward | 1690.62 | 1707.44 | 590.51 | 2.86x | 2.89x |
| fp32 1x3160x56x128 | backward | 2588.22 | 1775.28 | 803.46 | 3.22x | 2.21x |
| fp32 1x3160x56x128 | forward_backward | 4129.50 | 3253.62 | 1113.38 | 3.71x | 2.92x |
| bf16 1x73426x14x128 | forward | 5620.35 | 15524.24 | 2373.23 | 2.37x | 6.54x |
| bf16 1x73426x14x128 | backward | 7740.91 | 10866.53 | 2622.50 | 2.95x | 4.14x |
| bf16 1x73426x14x128 | forward_backward | 13195.31 | 26268.93 | 4712.98 | 2.80x | 5.57x |
| fp32 1x73426x14x128 | forward | 9609.38 | 9634.30 | 3400.74 | 2.83x | 2.83x |
| fp32 1x73426x14x128 | backward | 14304.48 | 9358.29 | 3716.03 | 3.85x | 2.52x |
| fp32 1x73426x14x128 | forward_backward | 23726.51 | 18695.22 | 6774.22 | 3.50x | 2.76x |
| bf16 1x73426x56x128 | forward | 22734.30 | 61117.97 | 9322.98 | 2.44x | 6.56x |
| bf16 1x73426x56x128 | backward | 30979.41 | 42287.06 | 9768.26 | 3.17x | 4.33x |
| bf16 1x73426x56x128 | forward_backward | 53475.97 | 103180.15 | 18603.33 | 2.87x | 5.55x |
| fp32 1x73426x56x128 | forward | 38202.24 | 38253.57 | 13397.22 | 2.85x | 2.86x |
| fp32 1x73426x56x128 | backward | 56603.02 | 36566.16 | 13846.75 | 4.09x | 2.64x |
| fp32 1x73426x56x128 | forward_backward | 94476.00 | 74366.64 | 26727.34 | 3.53x | 2.78x |

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
