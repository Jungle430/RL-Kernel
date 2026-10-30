# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Hardware identities from caller-provided metadata, without device initialization."""


def is_h100(
    *, device_type: str, device_name: str, capability: tuple[int, int], is_rocm: bool
) -> bool:
    """Recognize the NVIDIA H100 scope without querying or initializing a GPU.

    A launcher supplies metadata for the input device, not the current device.
    SM90 alone also matches H200; this release only qualifies H100. Eligibility
    is not a performance dispatch rule: the enclosing operator must qualify
    its dtype/shape and provide both Gluon and default Triton kernels.
    """
    return (
        device_type == "cuda"
        and not is_rocm
        and capability == (9, 0)
        and "H100" in device_name.upper().split()
    )
