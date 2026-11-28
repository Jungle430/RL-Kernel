# H3 partial MM-RoPE benchmark

Eager public wrappers; event medians; no torch.compile or CUDA Graphs. Backward reuses a graph; forward+backward creates a fresh graph. Inputs, table generation, compilation and checks are outside timing.

Provider = original model expression/autograd. Native = matched forward with explicit FP32 backward. BF16 provider backward rounds products earlier, so its gradient is not required to be bitwise equal.

Environment: `{'python': '3.12.3', 'torch': '2.8.0+cu128', 'triton': '3.4.0', 'cuda': '12.8', 'hip': None, 'gpu': 'NVIDIA A40', 'compute_capability': (8, 6), 'cpu_threads': 1}`

Complete: True

Acceptance: model coefficient/product storage boundaries, independent FP32 VJP. Unrounded FP32 mathematical error is diagnostic only; its outlier counts below are not waived or interpreted as a pure FP32 accuracy pass.

| Dtype / shape | Mode | Provider us | Native us | Triton us | Provider / Triton | Native / Triton |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bf16 1x264x14x128 | forward | 157.92 | 277.57 | 107.50 | 1.47x | 2.58x |
| bf16 1x264x14x128 | backward | 830.77 | 1050.72 | 592.62 | 1.40x | 1.77x |
| bf16 1x264x14x128 | forward_backward | 1643.20 | 2168.32 | 1163.54 | 1.41x | 1.86x |
| fp32 1x264x14x128 | forward | 261.90 | 316.80 | 197.89 | 1.32x | 1.60x |
| fp32 1x264x14x128 | backward | 984.85 | 986.34 | 746.50 | 1.32x | 1.32x |
| fp32 1x264x14x128 | forward_backward | 1224.03 | 1283.89 | 842.24 | 1.45x | 1.52x |
| bf16 1x264x56x128 | forward | 186.26 | 352.96 | 123.63 | 1.51x | 2.85x |
| bf16 1x264x56x128 | backward | 445.41 | 585.57 | 288.11 | 1.55x | 2.03x |
| bf16 1x264x56x128 | forward_backward | 764.70 | 1048.11 | 468.13 | 1.63x | 2.24x |
| fp32 1x264x56x128 | forward | 212.54 | 240.50 | 131.36 | 1.62x | 1.83x |
| fp32 1x264x56x128 | backward | 456.08 | 422.11 | 267.62 | 1.70x | 1.58x |
| fp32 1x264x56x128 | forward_backward | 1389.14 | 1411.04 | 1057.63 | 1.31x | 1.33x |
| bf16 1x925x14x128 | forward | 168.53 | 312.78 | 114.91 | 1.47x | 2.72x |
| bf16 1x925x14x128 | backward | 650.38 | 742.94 | 505.06 | 1.29x | 1.47x |
| bf16 1x925x14x128 | forward_backward | 787.76 | 1064.14 | 579.55 | 1.36x | 1.84x |
| fp32 1x925x14x128 | forward | 190.34 | 215.52 | 115.39 | 1.65x | 1.87x |
| fp32 1x925x14x128 | backward | 622.62 | 587.01 | 488.00 | 1.28x | 1.20x |
| fp32 1x925x14x128 | forward_backward | 1210.24 | 1266.18 | 927.02 | 1.31x | 1.37x |
| bf16 1x925x56x128 | forward | 363.47 | 900.21 | 146.02 | 2.49x | 6.17x |
| bf16 1x925x56x128 | backward | 713.71 | 982.53 | 482.45 | 1.48x | 2.04x |
| bf16 1x925x56x128 | forward_backward | 787.95 | 1447.49 | 466.22 | 1.69x | 3.10x |
| fp32 1x925x56x128 | forward | 546.35 | 566.85 | 181.33 | 3.01x | 3.13x |
| fp32 1x925x56x128 | backward | 944.98 | 792.61 | 517.23 | 1.83x | 1.53x |
| fp32 1x925x56x128 | forward_backward | 1336.19 | 1074.06 | 676.24 | 1.98x | 1.59x |
| bf16 1x3160x14x128 | forward | 329.70 | 795.50 | 152.37 | 2.16x | 5.22x |
| bf16 1x3160x14x128 | backward | 832.74 | 1184.38 | 615.44 | 1.35x | 1.92x |
| bf16 1x3160x14x128 | forward_backward | 1371.30 | 1804.83 | 895.25 | 1.53x | 2.02x |
| fp32 1x3160x14x128 | forward | 495.46 | 520.69 | 228.45 | 2.17x | 2.28x |
| fp32 1x3160x14x128 | backward | 951.70 | 942.64 | 718.53 | 1.32x | 1.31x |
| fp32 1x3160x14x128 | forward_backward | 1272.06 | 1103.65 | 798.11 | 1.59x | 1.38x |
| bf16 1x3160x56x128 | forward | 991.42 | 2708.85 | 337.06 | 2.94x | 8.04x |
| bf16 1x3160x56x128 | backward | 1434.78 | 2003.06 | 455.71 | 3.15x | 4.40x |
| bf16 1x3160x56x128 | forward_backward | 2346.86 | 4542.91 | 654.02 | 3.59x | 6.95x |
| fp32 1x3160x56x128 | forward | 1710.40 | 1733.76 | 621.41 | 2.75x | 2.79x |
| fp32 1x3160x56x128 | backward | 2557.76 | 1694.38 | 702.72 | 3.64x | 2.41x |
| fp32 1x3160x56x128 | forward_backward | 4156.80 | 3283.12 | 1155.17 | 3.60x | 2.84x |
| bf16 1x73426x14x128 | forward | 5623.74 | 15543.46 | 2393.14 | 2.35x | 6.50x |
| bf16 1x73426x14x128 | backward | 7804.50 | 10931.41 | 2712.00 | 2.88x | 4.03x |
| bf16 1x73426x14x128 | forward_backward | 13229.10 | 26285.07 | 4719.68 | 2.80x | 5.57x |
| fp32 1x73426x14x128 | forward | 9605.82 | 9653.47 | 3455.90 | 2.78x | 2.79x |
| fp32 1x73426x14x128 | backward | 14310.21 | 9316.03 | 3635.98 | 3.94x | 2.56x |
| fp32 1x73426x14x128 | forward_backward | 23780.21 | 18761.30 | 6852.64 | 3.47x | 2.74x |
| bf16 1x73426x56x128 | forward | 22772.66 | 61144.34 | 9353.78 | 2.43x | 6.54x |
| bf16 1x73426x56x128 | backward | 31012.10 | 42416.39 | 9834.35 | 3.15x | 4.31x |
| bf16 1x73426x56x128 | forward_backward | 53431.17 | 103171.23 | 18685.01 | 2.86x | 5.52x |
| fp32 1x73426x56x128 | forward | 38235.38 | 38269.97 | 13473.26 | 2.84x | 2.84x |
| fp32 1x73426x56x128 | backward | 56612.64 | 36567.58 | 13827.98 | 4.09x | 2.64x |
| fp32 1x73426x56x128 | forward_backward | 94608.32 | 74481.90 | 26812.00 | 3.53x | 2.78x |

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
