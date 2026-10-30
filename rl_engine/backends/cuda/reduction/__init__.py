# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""H100 ordered reductions embedded in an enclosing @gluon.jit kernel.

The six operations mirror the default Triton package's arithmetic contract.
The input is a loaded Gluon vector or scalar pointer; results are device scalars.
There are no Tensor wrappers, allocations, standalone kernels or automatic
hardware dispatch here. The caller selects its entire enclosing kernel on the
host. Ordinary @triton.jit callers continue using the shared Triton package.
"""

from .gluon import product, product_partials, reduce, reduce_partials, sum, sum_partials

__all__ = ["sum", "product", "sum_partials", "product_partials", "reduce", "reduce_partials"]
