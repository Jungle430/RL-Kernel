# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Eager references for the Triton ordered-reduction package exports.

sum/product default to loaded-vector semantics: a power-of-two last dimension,
optional suffix masking via length, and no extra seed. from_memory=True models
Triton's scalar-pointer path with fixed tiles and an ascending seeded fold.
Torch tensors cannot otherwise distinguish loaded values from pointer storage.
compute_dtype=None retains input precision; leading dimensions batch reductions.

sum_partials/product_partials merge stored tile results from zero/one in order.
Run eagerly; use no_grad for gradient-requiring inputs. These are fixed-path
references, separate from independent native or higher-precision accuracy checks.

reduce/reduce_partials expose the same implementations with caller-owned binary
functions. sum/product and their partial helpers supply standard operations and
neutral values. Custom operations need no registry or implementation-module import.
"""

from .ordered_reduce import product, product_partials, reduce, reduce_partials, sum, sum_partials

__all__ = ["sum", "product", "sum_partials", "product_partials", "reduce", "reduce_partials"]
