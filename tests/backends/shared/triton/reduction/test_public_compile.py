# SPDX-License-Identifier: Apache-2.0
"""Compile public helpers without a GPU to catch JIT-only argument-binding errors."""

import pytest


@pytest.fixture(scope="module")
def cuda_compile():
    """Build CUDA kernels for a fixed target, without allocating or launching them.

    Python/interpreter execution accepts keyword-only arguments that Triton's
    JIT frontend may not bind. Exercise the real compiler even on a CPU host
    with Triton installed; a missing runtime is an explicit skip.
    """
    triton = pytest.importorskip("triton")
    if not hasattr(triton, "compile") or not hasattr(triton, "constexpr_function"):
        pytest.skip("A complete Triton runtime with constexpr_function is required")
    from triton.backends import backends
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    if "nvidia" not in backends:
        pytest.skip("Triton's NVIDIA compilation backend is required")

    def compile_probe(probe, constants, pointer_type):
        signature = {
            name: "constexpr" if name in constants else pointer_type for name in probe.arg_names
        }
        source = ASTSource(probe, signature=signature, constexprs=constants)
        compiled = triton.compile(
            source,
            target=GPUTarget("cuda", 86, 32),
            options={"num_warps": 4, "enable_fp_fusion": False},
        )
        assert compiled.asm["ptx"]

    return compile_probe


@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("input_form", ("loaded", "pointer", "partials"))
def test_public_defaults_and_dtype_overrides_compile(cuda_compile, operation, input_form):
    """Compile omitted/None/explicit dtype calls through all convenience wrappers."""
    from rl_engine.backends.shared.triton import reduction
    from tests.helpers.ordered_reduce_kernels import (
        inferred_tile_dtype_probe,
        inferred_vector_dtype_probe,
        public_partials_probe,
    )

    if input_form == "loaded":
        probe = inferred_tile_dtype_probe
        constants = {"BLOCK_SIZE": 64, "REDUCE_FN": getattr(reduction, operation)}
    elif input_form == "pointer":
        probe = inferred_vector_dtype_probe
        constants = {"LENGTH": 33, "REDUCE_FN": getattr(reduction, operation)}
    else:
        probe = public_partials_probe
        constants = {
            "N_TILES": 5,
            "REDUCE_FN": getattr(reduction, f"{operation}_partials"),
            "MERGE_UNROLL": 3,
        }
    cuda_compile(probe, constants, "*fp32")


@pytest.mark.parametrize("input_form", ("input", "partials"))
def test_custom_callback_and_runtime_seed_compile(cuda_compile, input_form):
    """Compile caller-owned callbacks with default precision and a loaded merge seed."""
    from tests.helpers.ordered_reduce_kernels import (
        boolean_xor,
        custom_input_probe,
        custom_merge_probe,
    )

    if input_form == "input":
        probe = custom_input_probe
        constants = {"WIDTH": 64, "LENGTH": 33, "COMBINE_FN": boolean_xor, "IDENTITY": 0}
    else:
        probe = custom_merge_probe
        constants = {"N_TILES": 5, "COMBINE_FN": boolean_xor, "MERGE_UNROLL": 3}
    cuda_compile(probe, constants, "*i64")
