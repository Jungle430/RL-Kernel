# SPDX-License-Identifier: Apache-2.0
"""Hardware eligibility uses supplied metadata, without querying an accelerator."""

import pytest

from rl_engine.platforms.hardware import is_h100


@pytest.mark.parametrize(
    "name,capability,rocm,expected",
    [
        ("NVIDIA H100 80GB HBM3", (9, 0), False, True),
        ("NVIDIA H100 PCIe", (9, 0), False, True),
        ("NVIDIA H200", (9, 0), False, False),
        ("NVIDIA A100", (8, 0), False, False),
        ("NVIDIA H100 80GB HBM3", (8, 0), False, False),
        ("NVIDIA H100 80GB HBM3", (9, 0), True, False),
    ],
)
def test_h100_scope_is_metadata_only(name, capability, rocm, expected):
    assert (
        is_h100(device_type="cuda", device_name=name, capability=capability, is_rocm=rocm)
        == expected
    )
    assert not is_h100(device_type="cpu", device_name=name, capability=capability, is_rocm=rocm)
