# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Device-local fixed-order reductions and sum/product helpers embedded inside Triton kernels.

sum/product dispatch on the compile-time input type: loaded vectors keep their
existing tile tree; scalar pointers use fixed tiles and an ascending seeded fold.
length selects a prefix in either path, never whether memory is loaded.
sum_partials/product_partials merge stored producer results without a new tree.

Public reduce/reduce_partials accept custom JIT callbacks. The generic tree,
partial, serial and merge stages share arithmetic with custom
fused/parallel operators. The combiner, padding identity and final seed remain
separate. Elementwise transforms belong to the caller. There is no dtype
whitelist or implicit widening: callers select supported arithmetic precision
and own allocation, launch/device/stream ordering and compiler math settings.
Fixed order is not exact arithmetic or a cross-hardware reproducibility promise.
Independent accuracy and GPU conformance tests remain necessary.
"""

from types import SimpleNamespace

import triton
import triton.language as tl

from rl_engine.contracts.operators.ordered_reduction import MAX_GROUP_SIZE, SECTION_COUNT, TILE_SIZE


@triton.jit
def reduce_add(left, right):
    """Add two operands; zero is the padding identity."""
    return left + right


@triton.jit
def reduce_multiply(left, right):
    """Multiply two operands; one is the padding identity."""
    return left * right


@triton.constexpr_function
def _reduction_callback(combine_fn):
    """Hold the caller's JIT combiner on a plain Python object at compile time.

    Triton 3.7.1 passes a constexpr wrapper for a JIT function parameter, but
    tl.reduce expects the original JIT function. Reading constexpr.value
    directly makes the compiler bind that function as a method instead.
    constexpr_function unwraps its arguments; the plain object's attribute
    then exposes the original function without Triton's method binding.

    This namespace is never a GPU value. It adds no device allocation,
    arithmetic or dispatch, and also works with interpreter-mode functions.
    """
    return SimpleNamespace(combine=combine_fn)


@triton.jit
def ordered_tile_reduce(
    values,
    MAX_GROUP_SIZE: tl.constexpr,
    SECTION_COUNT: tl.constexpr,
    COMBINE_FN: tl.constexpr,
):
    """Combine one already-computed Triton vector into a scalar in a fixed order.

    Call this helper inside a Triton kernel after loading or computing a tile.
    It reduces exactly the supplied values: reduce_add gives their sum, while
    reduce_multiply gives their product. The grouping parameters determine
    which operands combine at every step, independently of tile scheduling.
    Loading, elementwise transforms and masking belong to the caller; this
    function neither reads pointers nor launches another kernel.

    Args:
        values: One-dimensional Triton tensor with shape [BLOCK_SIZE], where
            BLOCK_SIZE is a positive power of two. These are the actual values
            to reduce, not a pointer. Any padded lanes must already contain the
            operation's neutral value, such as zero for sum or one for product.
        MAX_GROUP_SIZE: Compile-time maximum number of adjacent elements in
            each initial group; must be a positive power of two. The actual
            group size is min(BLOCK_SIZE, MAX_GROUP_SIZE). Each group combines
            left to right. These are arithmetic groups, not GPU thread groups.
        SECTION_COUNT: Compile-time maximum number of equal sections of group
            totals; must be a positive power of two. The actual section count
            is min(N_GROUPS, SECTION_COUNT), so no empty sections are inserted.
            Each section is reduced first, then its total joins the section tree.
        COMBINE_FN: Compile-time-selected, pure @triton.jit binary function,
            called as COMBINE_FN(left, right). It must return the same shape
            and dtype as its operands. Choose a mathematically associative,
            commutative operation; floating-point rounding may still depend
            on the prescribed order. The combiner owns any arithmetic casts.

    Returns:
        A scalar Triton tensor with shape () and the same dtype as values,
        ready for further computation or storage by the enclosing kernel.

    Reduction order:
        1. Split values into adjacent groups and fold each group from its first
           element, without inserting an initial value.
        2. Divide group totals into min(N_GROUPS, SECTION_COUNT) equal sections.
           Within each section, repeatedly combine corresponding entries from
           its lower and upper halves until one section total remains.
        3. Repeatedly combine the lower and upper halves of the section totals.
           Four sections retain combine(combine(s0, s2), combine(s1, s3)).
           A single entry at either stage passes through without a combine.

    Every tree step combines only the prescribed left/right pair through a
    two-entry reduction, for both built-in and caller-owned JIT combiners.
    Each reshape fixes the reduced axis at length two. This exposes the pair
    to Triton's reduction lowering without requiring permute/split layouts
    or letting the compiler choose a larger reduction tree.
    No seed, dtype conversion or merge across separate tiles is added here.
    The same BLOCK_SIZE, MAX_GROUP_SIZE and SECTION_COUNT determine the same
    operand tree. Bitwise conformance also requires identical inputs, combiner,
    dtype and rounding behavior. Changing the arithmetic configuration can
    change results; fixed order alone does not promise equal bits across
    backends or compiler versions.

    Example:
        tile_sum = ordered_tile_reduce(
            exponentials,
            MAX_GROUP_SIZE=8,
            SECTION_COUNT=4,
            COMBINE_FN=reduce_add,
        )
    """
    tl.static_assert(len(values.shape) == 1, "ordered tile reduction requires a vector")
    BLOCK_SIZE: tl.constexpr = values.shape[0]
    tl.static_assert(
        BLOCK_SIZE > 0 and (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0,
        "ordered tile width must be a positive power of two",
    )
    tl.static_assert(
        MAX_GROUP_SIZE > 0 and (MAX_GROUP_SIZE & (MAX_GROUP_SIZE - 1)) == 0,
        "maximum group size must be a positive power of two",
    )
    tl.static_assert(
        SECTION_COUNT > 0 and (SECTION_COUNT & (SECTION_COUNT - 1)) == 0,
        "section count must be a positive power of two",
    )
    GROUP_SIZE: tl.constexpr = min(BLOCK_SIZE, MAX_GROUP_SIZE)
    N_GROUPS: tl.constexpr = BLOCK_SIZE // GROUP_SIZE
    # Keep indexed extraction: split-based extraction caused LLVM O3 compilation
    # timeouts with three-stage pipelining on H100/Triton 3.7.1. The group fold
    # below retains the declared operand order without requiring that layout.
    groups = tl.arange(0, N_GROUPS) * GROUP_SIZE
    subtotal = tl.gather(values, groups, axis=0)
    for offset in tl.static_range(1, GROUP_SIZE):
        subtotal = COMBINE_FN(subtotal, tl.gather(values, groups + offset, axis=0))
    N_SECTIONS: tl.constexpr = min(N_GROUPS, SECTION_COUNT)
    SECTION_SIZE: tl.constexpr = N_GROUPS // N_SECTIONS
    # Convert only compile-time sizes: the interpreter can retain constexpr
    # wrappers, which do not expose Python int.bit_length directly.
    # Each reduced axis has exactly two entries: the prescribed lower/upper pair.
    for shift in tl.static_range(int(SECTION_SIZE).bit_length() - 2, -1, -1):
        pairs = tl.reshape(subtotal, (N_SECTIONS, 2, 1 << shift))
        subtotal = tl.reduce(pairs, axis=1, combine_fn=_reduction_callback(COMBINE_FN).combine)
    subtotal = tl.reshape(subtotal, (N_SECTIONS,))
    # Merge section totals by halving distance: four sections pair 0+2, 1+3.
    for shift in tl.static_range(int(N_SECTIONS).bit_length() - 2, -1, -1):
        pairs = tl.reshape(subtotal, (2, 1 << shift))
        subtotal = tl.reduce(pairs, axis=0, combine_fn=_reduction_callback(COMBINE_FN).combine)
    return tl.reshape(subtotal, ())


@triton.jit
def ordered_reduce_partial(
    input_ptr,
    length: tl.constexpr,
    tile_index,
    BLOCK_SIZE: tl.constexpr,
    MAX_GROUP_SIZE: tl.constexpr,
    SECTION_COUNT: tl.constexpr,
    COMBINE_FN: tl.constexpr,
    IDENTITY: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """Load and reduce one tile of a contiguous vector in COMPUTE_DTYPE.

    input_ptr addresses the vector's beginning; tile_index must be nonnegative.
    Cast loaded values to the
    requested dtype, then fill padded lanes with IDENTITY in that dtype.
    IDENTITY must be neutral for COMBINE_FN on the supported value domain.
    No user seed enters partials. Both schedules use this exact operation.
    To fuse a transform, compute it in the caller and use ordered_tile_reduce.
    """
    offsets = tl.cast(tile_index, tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < length
    values = tl.load(input_ptr + offsets, mask=mask, other=0).to(COMPUTE_DTYPE)
    padding = tl.full((), IDENTITY, COMPUTE_DTYPE)
    values = tl.where(mask, values, padding)
    return ordered_tile_reduce(
        values,
        MAX_GROUP_SIZE=MAX_GROUP_SIZE,
        SECTION_COUNT=SECTION_COUNT,
        COMBINE_FN=COMBINE_FN,
    )


@triton.jit
def ordered_reduce_serial(
    input_ptr,
    length: tl.constexpr,
    initial,
    BLOCK_SIZE: tl.constexpr,
    MAX_GROUP_SIZE: tl.constexpr,
    SECTION_COUNT: tl.constexpr,
    COMBINE_FN: tl.constexpr,
    IDENTITY: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    """Reduce a contiguous vector in one program, one tile at a time.

    Fold tile results left to right, starting with initial converted to COMPUTE_DTYPE.
    initial is applied once, not once per tile, and may be a runtime scalar.
    IDENTITY is only for padding; it can differ from initial. The matching
    parallel merge receives the same initial and COMBINE_FN. PIPELINE_STAGES
    changes scheduling, not combine pairs. Length zero returns initial.
    No global partial buffer is needed. Elementwise transforms are not applied.
    """
    tl.static_assert(length >= 0, "vector length must be nonnegative")
    tl.static_assert(
        BLOCK_SIZE > 0 and (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0,
        "block size must be a positive power of two",
    )
    tl.static_assert(
        MAX_GROUP_SIZE > 0 and (MAX_GROUP_SIZE & (MAX_GROUP_SIZE - 1)) == 0,
        "maximum group size must be a positive power of two",
    )
    tl.static_assert(
        SECTION_COUNT > 0 and (SECTION_COUNT & (SECTION_COUNT - 1)) == 0,
        "section count must be a positive power of two",
    )
    n_tiles: tl.constexpr = (length + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Construct constants directly in the chosen dtype; a scalar cast can first
    # materialize a Python float as FP32 and lose bits before conversion to FP64.
    total = tl.full((), initial, COMPUTE_DTYPE)
    for tile in tl.range(0, n_tiles, num_stages=PIPELINE_STAGES):
        partial = ordered_reduce_partial(
            input_ptr,
            length,
            tile,
            BLOCK_SIZE=BLOCK_SIZE,
            MAX_GROUP_SIZE=MAX_GROUP_SIZE,
            SECTION_COUNT=SECTION_COUNT,
            COMBINE_FN=COMBINE_FN,
            IDENTITY=IDENTITY,
            COMPUTE_DTYPE=COMPUTE_DTYPE,
        )
        total = COMBINE_FN(total, partial)
    return total


@triton.jit
def ordered_reduce_merge(
    partials_ptr,
    n_tiles: tl.constexpr,
    initial,
    COMBINE_FN: tl.constexpr,
    MERGE_UNROLL: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """Fold unseeded COMPUTE_DTYPE partials in ascending tile order, starting at initial.

    A parallel producer stores tile k at partials_ptr+k. Its completion order
    does not select the reduction order. Apply initial once, before tile zero,
    exactly as the serial schedule does. MERGE_UNROLL is any positive integer:
    it expands that many load/combine steps per loop iteration at compile time.
    It need not be a power of two. Combines remain left-associated, including
    incomplete tails; changing the unroll factor does not change their order.
    This is not a balanced tree over partials. No padding/identity is needed
    in this stage. Zero partials return initial without a load or combine.

    The caller must complete all producers before reading partials, using the
    same stream or an explicit dependency. This helper does not synchronize.
    Use this fold for stored partials; sum would introduce a new tile tree.
    """
    tl.static_assert(n_tiles >= 0, "partial count must be nonnegative")
    total = tl.full((), initial, COMPUTE_DTYPE)
    tl.static_assert(
        partials_ptr.dtype.element_ty == COMPUTE_DTYPE, "partial storage must match compute dtype"
    )
    tl.static_assert(MERGE_UNROLL > 0, "merge unroll must be a positive integer")
    full_tiles: tl.constexpr = (n_tiles // MERGE_UNROLL) * MERGE_UNROLL
    for tile in range(0, full_tiles, MERGE_UNROLL):
        for offset in tl.static_range(MERGE_UNROLL):
            partial = tl.load(partials_ptr + tile + offset)
            total = COMBINE_FN(total, partial)
    for tile in tl.static_range(full_tiles, n_tiles):
        total = COMBINE_FN(total, tl.load(partials_ptr + tile))
    return total


@triton.jit
def reduce(
    input,
    length: tl.constexpr = None,
    *,
    COMBINE_FN: tl.constexpr,
    IDENTITY: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr = None,
    BLOCK_SIZE: tl.constexpr = TILE_SIZE,
    MAX_GROUP_SIZE: tl.constexpr = MAX_GROUP_SIZE,
    SECTION_COUNT: tl.constexpr = SECTION_COUNT,
    PIPELINE_STAGES: tl.constexpr = None,
):
    """Reduce loaded values or a memory vector with a caller-owned binary function.

    This is the shared public implementation used by sum/product. Supply only
    the operation and its neutral value; arithmetic and loading options have
    the same defaults as sum. No registry or built-in-operation switch is used.
    Call from another @triton.jit function, not as a standalone kernel launch.

    Args:
        input: Loaded one-dimensional numeric values with positive power-of-two
            width, or a scalar pointer to contiguous device memory.
        length: Optional compile-time prefix length for loaded values; required
            for pointers. Loaded lengths must lie between zero and their width.
            Input type selects the path; length never triggers a memory load.
        COMBINE_FN: Pure @triton.jit function called as fn(left, right). It must
            preserve shape and dtype and be mathematically associative and
            commutative on the supported domain. Floating-point rounding can
            still depend on the prescribed pairs. It must not load, store or
            mutate state. This interface returns one scalar, not a tuple state.
        IDENTITY: Compile-time neutral value satisfying fn(x, e) = fn(e, x) = x
            on that domain, e.g. 0 for addition/XOR or 1 for multiplication.
            Converted directly into the compute dtype, it fills excluded lanes
            and initializes the pointer path's outer fold. It is not added as
            a seed to loaded tiles. The caller is responsible for its validity.
        COMPUTE_DTYPE: None retains the numeric dtype or pointer element type.
            An explicit dtype converts inputs before applying the combiner.
        BLOCK_SIZE: Pointer-only arithmetic tile width, default 1024.
        MAX_GROUP_SIZE: Maximum adjacent group size, default 8.
        SECTION_COUNT: Maximum section count, default 4. All three sizes must
            be positive powers of two. Loaded values retain their existing width.
        PIPELINE_STAGES: Pointer-only software-pipeline hint; None is unhinted.
            It changes scheduling, not the order of combines.

    Loaded values follow the seedless tile tree. Optional prefix masking retains
    their width. Pointers load fixed tiles, then fold unseeded partials in ascending
    order from IDENTITY. Empty prefixes return IDENTITY. Result dtype is the
    compute dtype; there is no automatic widening, allocation, store or launch.
    Callers own math compiler flags and producer/merge dependencies. Matching
    bits requires the same tree, dtype, callback and inputs, not just the name.
    Subtraction/division cannot be substituted for associative reductions with a
    neutral padding value. For custom parallel producers, merge their unseeded
    results using reduce_partials with the same callback and initial=IDENTITY.

    Example:
        @triton.jit
        def xor_pair(left, right):
            return left ^ right

        flags = reduce(values, COMBINE_FN=xor_pair, IDENTITY=0)
        flags = reduce(ptr, length=n, COMBINE_FN=xor_pair, IDENTITY=0)
    """
    input_dtype: tl.constexpr = input.dtype.element_ty if input.dtype.is_ptr() else input.dtype
    compute_dtype: tl.constexpr = input_dtype if COMPUTE_DTYPE is None else COMPUTE_DTYPE
    if input.dtype.is_ptr():
        tl.static_assert(len(input.shape) == 0, "input must be a scalar pointer")
        tl.static_assert(length is not None, "pointer input requires length")
        return ordered_reduce_serial(
            input,
            length,
            initial=IDENTITY,
            BLOCK_SIZE=BLOCK_SIZE,
            MAX_GROUP_SIZE=MAX_GROUP_SIZE,
            SECTION_COUNT=SECTION_COUNT,
            COMBINE_FN=COMBINE_FN,
            IDENTITY=IDENTITY,
            COMPUTE_DTYPE=compute_dtype,
            PIPELINE_STAGES=PIPELINE_STAGES,
        )
    else:
        tl.static_assert(len(input.shape) == 1, "loaded input must be a vector")
        values = input.to(compute_dtype)
        if length is not None:
            tl.static_assert(
                0 <= length and length <= input.shape[0],
                "loaded length must be between zero and the vector width",
            )
            values = tl.where(
                tl.arange(0, input.shape[0]) < length,
                values,
                tl.full((), IDENTITY, compute_dtype),
            )
        return ordered_tile_reduce(
            values,
            MAX_GROUP_SIZE=MAX_GROUP_SIZE,
            SECTION_COUNT=SECTION_COUNT,
            COMBINE_FN=COMBINE_FN,
        )


@triton.jit
def sum(
    input,
    length: tl.constexpr = None,
    *,
    COMPUTE_DTYPE: tl.constexpr = None,
    BLOCK_SIZE: tl.constexpr = TILE_SIZE,
    MAX_GROUP_SIZE: tl.constexpr = MAX_GROUP_SIZE,
    SECTION_COUNT: tl.constexpr = SECTION_COUNT,
    PIPELINE_STAGES: tl.constexpr = None,
):
    """Sum loaded values or a contiguous memory vector in a prescribed order.

    Call inside another @triton.jit function. Input type selects the path at
    compile time; length only specifies the participating prefix:

        sum(values)                 # Reduce all already-loaded values.
        sum(values, length=128)     # Mask the loaded suffix with zero.
        sum(input_ptr, length=2500) # Load and reduce successive fixed-width tiles.

    Args:
        input: Either a one-dimensional numeric vector whose width is a positive
            power of two, or a scalar pointer to contiguous device memory.
            Loaded values may be intermediate results computed by the caller.
            A vector of pointers is not accepted; load it in the caller first.
        length: Optional nonnegative compile-time prefix length. Required for
            pointers; omitted for loaded values means their full width. For
            loaded values it must not exceed that width. Zero yields zero.
        COMPUTE_DTYPE: None uses the loaded dtype or pointer element type.
            An explicit dtype converts input before arithmetic. There is no
            automatic widening; choose an integer dtype to count booleans.
        BLOCK_SIZE: Pointer path only: positive power-of-two arithmetic tile
            width, default 1024. The total length may be larger or non-power-of-two.
            Loaded input retains its existing width regardless of this option.
        MAX_GROUP_SIZE: Positive power-of-two maximum adjacent group size,
            default 8. Each group accumulates sequentially from its first value.
        SECTION_COUNT: Positive power-of-two maximum section count, default 4.
            Group totals follow the fixed lower-half/upper-half tree.
        PIPELINE_STAGES: Pointer path only: optional tl.range loop-pipeline hint.
            None leaves it unhinted. Scheduling does not change combine order.

    Returns a scalar in the compute dtype; no store, allocation or kernel launch
    occurs. The pointer path pads the final tile with zero, then adds tile sums
    left to right from positive zero. The loaded path adds no seed and masks
    any excluded suffix with zero without shrinking or retiling the vector.
    Thus sum(values, length=n) need not be bitwise equal to sum(ptr, length=n):
    loaded width, pointer tile width and the outer seed must also be considered.
    Keep tree parameters, precision and compiler math settings fixed across
    paths requiring matching bits. Fixed order is not exact arithmetic.
    For an arbitrary loaded subset, mask unwanted lanes with zero in the caller.
    To merge stored tile sums, use sum_partials instead of introducing a new tree.
    """
    return reduce(
        input,
        length,
        COMBINE_FN=reduce_add,
        IDENTITY=0,
        COMPUTE_DTYPE=COMPUTE_DTYPE,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_GROUP_SIZE=MAX_GROUP_SIZE,
        SECTION_COUNT=SECTION_COUNT,
        PIPELINE_STAGES=PIPELINE_STAGES,
    )


@triton.jit
def product(
    input,
    length: tl.constexpr = None,
    *,
    COMPUTE_DTYPE: tl.constexpr = None,
    BLOCK_SIZE: tl.constexpr = TILE_SIZE,
    MAX_GROUP_SIZE: tl.constexpr = MAX_GROUP_SIZE,
    SECTION_COUNT: tl.constexpr = SECTION_COUNT,
    PIPELINE_STAGES: tl.constexpr = None,
):
    """Multiply loaded values or a contiguous memory vector in a prescribed order.

    Mirrors sum's input dispatch and configuration. product(values) reduces a
    loaded power-of-two vector; product(values, length=n) replaces its suffix
    with one, preserving the original width and seedless tree. Arbitrary loaded
    subsets can be selected by replacing unwanted lanes with one in the caller.
    product(ptr, length=n) loads fixed BLOCK_SIZE tiles, pads the tail with one,
    then multiplies partials left to right starting from one. Empty prefixes
    return one. Pointer inputs require length; loaded prefixes cannot exceed
    the vector width. All lengths are compile-time values.

    COMPUTE_DTYPE=None retains the input dtype, without widening. BLOCK_SIZE
    (default 1024) and PIPELINE_STAGES (default None) affect only pointer input.
    MAX_GROUP_SIZE (default 8) and SECTION_COUNT (default 4) control either tree;
    all arithmetic sizes must be positive powers of two. Input representation
    is selected at compile time, never by length or a runtime loading flag.
    The result is a scalar usable in surrounding fused arithmetic, with no
    allocation, store or launch. The caller owns parallel scheduling and streams.
    Fixed order does not prevent rounding, underflow or overflow. Use
    product_partials for stored tile products to preserve their left fold.
    """
    return reduce(
        input,
        length,
        COMBINE_FN=reduce_multiply,
        IDENTITY=1,
        COMPUTE_DTYPE=COMPUTE_DTYPE,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_GROUP_SIZE=MAX_GROUP_SIZE,
        SECTION_COUNT=SECTION_COUNT,
        PIPELINE_STAGES=PIPELINE_STAGES,
    )


@triton.jit
def reduce_partials(
    partials_ptr,
    n_tiles: tl.constexpr,
    *,
    COMBINE_FN: tl.constexpr,
    initial,
    COMPUTE_DTYPE: tl.constexpr = None,
    MERGE_UNROLL: tl.constexpr = 1,
):
    """Fold stored partials in ascending slot order with a custom binary function.

    partials_ptr is a scalar pointer to contiguous unseeded tile results;
    n_tiles is their nonnegative compile-time count. Each producer writes tile k
    to slot k. Return fn(...fn(fn(initial, p0), p1)...), with no new tile tree.
    An empty buffer returns initial. The callback has the same shape/dtype and
    purity requirements as reduce, and must match the producers' callback.

    initial is required and applied once before slot zero, never to each partial.
    To match reduce's pointer path, pass its IDENTITY. It may instead be a
    caller-selected seed, including a runtime scalar; it is distinct from the
    neutral value used to pad producer tiles. A merge has no padded lanes, so
    it needs no IDENTITY argument.

    COMPUTE_DTYPE=None infers partial storage dtype; an explicit dtype must match
    storage. MERGE_UNROLL defaults to 1; any positive integer preserves the fold.
    Producers must complete before this call via the same stream or an explicit
    dependency. This embedded helper does not allocate, launch or synchronize.
    """
    tl.static_assert(partials_ptr.dtype.is_ptr(), "partials must be a pointer")
    tl.static_assert(len(partials_ptr.shape) == 0, "partials must be a scalar pointer")
    compute_dtype: tl.constexpr = (
        partials_ptr.dtype.element_ty if COMPUTE_DTYPE is None else COMPUTE_DTYPE
    )
    return ordered_reduce_merge(
        partials_ptr,
        n_tiles,
        initial=initial,
        COMBINE_FN=COMBINE_FN,
        MERGE_UNROLL=MERGE_UNROLL,
        COMPUTE_DTYPE=compute_dtype,
    )


@triton.jit
def sum_partials(
    partials_ptr,
    n_tiles: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr = None,
    MERGE_UNROLL: tl.constexpr = 1,
):
    """Merge stored tile sums in order; only the pointer and count are required.

    Args:
        partials_ptr: Contiguous tile sums in ascending tile order. Each entry
            is an unseeded result computed by a producer.
        n_tiles: Nonnegative number of partials, known at compile time. It need
            not be a power of two; zero partials return positive zero.
        COMPUTE_DTYPE: None uses the partial storage dtype. An explicit dtype
            must match storage, so merging introduces no extra conversion.
        MERGE_UNROLL: Optional positive loop-unroll factor, defaulting to 1.
            It changes scheduling, not the addition order.

    Return (((0 + partial[0]) + partial[1]) + ...) in the partial storage dtype.
    This is the same fold used by sum's pointer path after reducing each tile, without
    applying another tile tree to the partials. The caller must complete all
    producers before this function runs; no kernel launch or synchronization
    occurs here.
    """
    return reduce_partials(
        partials_ptr,
        n_tiles,
        COMBINE_FN=reduce_add,
        initial=0,
        COMPUTE_DTYPE=COMPUTE_DTYPE,
        MERGE_UNROLL=MERGE_UNROLL,
    )


@triton.jit
def product_partials(
    partials_ptr,
    n_tiles: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr = None,
    MERGE_UNROLL: tl.constexpr = 1,
):
    """Merge stored tile products in order; only the pointer and count are required.

    Args:
        partials_ptr: Contiguous, unseeded tile products in ascending tile order.
        n_tiles: Nonnegative compile-time count; zero partials return one.
        COMPUTE_DTYPE: None uses the partial storage dtype. An explicit dtype
            must match storage, preserving the producer's arithmetic precision.
        MERGE_UNROLL: Optional positive loop-unroll factor, defaulting to 1;
            it does not change the multiplication order.

    Return (((1 * partial[0]) * partial[1]) * ...), matching product's pointer-path
    fold over tile results. Producers must finish before this function reads
    their results, as with sum_partials. No new tile tree or dtype cast is added.
    """
    return reduce_partials(
        partials_ptr,
        n_tiles,
        COMBINE_FN=reduce_multiply,
        initial=1,
        COMPUTE_DTYPE=COMPUTE_DTYPE,
        MERGE_UNROLL=MERGE_UNROLL,
    )
