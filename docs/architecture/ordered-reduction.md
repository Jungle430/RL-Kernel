# Device-local ordered reductions

Import the package and call a function inside your Triton kernel:

```python
import triton.language as tl
from rl_engine.backends.shared.triton import reduction as ordered

# Already-loaded/computed numeric values: no memory load inside the helper.
total = ordered.sum(values)
total = ordered.sum(values, length=128)  # Only the prefix participates.

# Scalar pointer: load and reduce a complete contiguous vector in fixed tiles.
total = ordered.sum(input_ptr, length=2500)
```

Input **type** selects the path at compile time. `length` only specifies a
participating prefix; passing it for loaded values does not make them pointers.
There is no `needLoad` flag. Four convenience functions cover sum and product;
two generic functions accept caller-owned binary operations:

| Export | Input | Result |
| --- | --- | --- |
| `sum` | Loaded values, optionally with a prefix length; or a scalar pointer and required length | Fixed-order sum |
| `product` | Same two input forms as `sum` | Fixed-order product |
| `sum_partials(ptr, n_tiles)` | Stored tile sums in canonical slot order | Left fold from zero |
| `product_partials(ptr, n_tiles)` | Stored tile products in canonical slot order | Left fold from one |
| `reduce` | Loaded values or pointer + length, with `COMBINE_FN` and `IDENTITY` | Custom operation using the same tree and input dispatch |
| `reduce_partials` | Stored partials, with `COMBINE_FN` and `initial` | Custom ascending left fold |

All return a scalar for surrounding fused computation or storage. They do not
allocate, store outputs, launch kernels or synchronize programs. Callers need
not provide a combiner or select an execution strategy for ordinary use.

## Loaded values and prefix masks

Loaded input is a one-dimensional numeric vector with positive power-of-two
width. `sum(values)` and `product(values)` reduce all of it without inserting
an extra seed. Optional `length` must be between zero and that width:

```python
# values has 1024 lanes; the helper masks lanes 128..1023 with zero.
total = ordered.sum(values, length=128)

# Product masks excluded lanes with one instead.
result = ordered.product(values, length=128)
```

The tree still has **1024 lanes**, not 128. Neither `length` nor the pointer-only
`BLOCK_SIZE` option shrinks or retiles loaded input. A zero-length prefix returns
zero for sum or one for product; a zero-width loaded vector is invalid.
Excluded NaNs are replaced by the neutral value before reduction. All lanes
are eligible when `length` is omitted, so the caller must mask other invalid lanes.

For an arbitrary subset, select participation in the caller:

```python
cols = tl.arange(0, 1024)
part = tl.where((cols >= 128) & (cols < 256), values, 0.0)
total = ordered.sum(part)
```

Elementwise transformations also belong in the caller. Passing computed values
directly avoids storing and reloading a temporary tensor. For product, replace
unwanted lanes with one rather than zero.

## Pointers, tiling and parallelism

Pointer input must be a scalar pointer to contiguous storage. Its nonnegative
compile-time `length` is required; it can exceed `BLOCK_SIZE` and need not be a
power of two. A vector of pointers must be loaded by the caller first.

```python
import triton

@triton.jit
def row_sum_kernel(x_ptr, out_ptr, V: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    total = ordered.sum(x_ptr + row * V, length=V, COMPUTE_DTYPE=tl.float32)
    tl.store(out_ptr + row, total)
```

Each program loads fixed-width tiles, pads the final tile with zero/one and
folds their results in ascending order starting from zero/one. Length zero
returns that initial value without loading. `BLOCK_SIZE` is an arithmetic tile
width, **not a limit on the complete vector length**.

The enclosing kernel's grid and `num_warps` determine program/thread parallelism.
The helper processes all tiles in its calling program; it does not split a
vector across multiple programs. Allocation, device/stream ordering and output
stores belong to the enclosing operator. There is no standalone Tensor launcher.

Pointer-only `PIPELINE_STAGES=None` leaves the `tl.range` tile loop unhinted.
An explicit value such as 3 changes software pipelining, not the tile tree or
partial fold. This is distinct from the enclosing kernel's launch `num_stages`.
Loaded input has no load loop, so this option has no effect on that path.

The two input forms retain their existing arithmetic contracts. Loaded input
uses its supplied width with **no extra seed**. Pointer input uses `BLOCK_SIZE`
and an outer seeded fold. Even identical valid values need not have identical
bits if those widths/seed semantics differ. A shared function name alone does
not establish bitwise equivalence between the paths.

## Merging stored partials

```python
# Inside a merge kernel after producers have completed:
total = ordered.sum_partials(partials_ptr, n_tiles)
result = ordered.product_partials(partials_ptr, n_tiles)
```

The partial helpers need only a pointer and count. Producers store tile k in
slot k, regardless of completion order. Merge performs an ascending left fold,
starting from zero/one, without adding another tile tree. Empty partials return
the seed. Optional positive `MERGE_UNROLL=1` changes loop expansion, not order.

`COMPUTE_DTYPE=None` uses the partial storage dtype; an explicit dtype must
match it. Producers must finish before the merge reads their output, using the
same stream or an explicit dependency. The helpers do not perform synchronization.

## Custom binary operations

The sum/product names are convenience specializations of the public `reduce`.
There is no callback registry or switch limited to built-in operations. For
example, integer XOR is available through a caller-owned JIT function:

```python
import triton
from rl_engine.backends.shared.triton import reduction as ordered

@triton.jit
def xor_pair(left, right):
    return left ^ right

# Inside another @triton.jit function:
flags = ordered.reduce(values, COMBINE_FN=xor_pair, IDENTITY=0)
flags = ordered.reduce(values, length=128, COMBINE_FN=xor_pair, IDENTITY=0)
flags = ordered.reduce(input_ptr, length=n, COMBINE_FN=xor_pair, IDENTITY=0)
```

`COMBINE_FN` determines what each pair does. `IDENTITY` supplies the neutral value
for masked lanes and the pointer path's initial accumulator; it cannot be
inferred from an arbitrary callback. Loaded tiles remain seedless. All other
input, length, compute-dtype and tree options are the same as for sum/product.

The callback must be pure, preserve shape/dtype, and represent a mathematically
associative and commutative operation on the supported domain. Floating-point
rounding still depends on the fixed tree. There is one scalar result: multi-field
states, arbitrary subtraction/division, loads/stores and side effects are outside
this callback contract. Elementwise maps belong in the caller.

For parallel producers, use the same callback at both stages:

```python
# Producer: caller has loaded/masked one fixed-width tile.
partial = ordered.reduce(tile, COMBINE_FN=xor_pair, IDENTITY=0)
# Caller stores partial at partials_ptr + tile_index.

# Dependent merge kernel, after all producers have completed:
flags = ordered.reduce_partials(
    partials_ptr, n_tiles, COMBINE_FN=xor_pair, initial=0,
)
```

`reduce_partials` does **not** build a new tree over the stored results. It folds
ascending slots, just like `sum_partials` and `product_partials`. The required
`initial` value enters once, before slot zero; empty partials return it. To match
`reduce`'s pointer path, use `initial=IDENTITY`. A caller may choose a different
seed or supply a runtime scalar. Initial accumulator and neutral padding have
different roles: the merge has no padded lanes, so it needs no `IDENTITY` argument.

The eager counterpart uses a plain PyTorch binary function:

```python
import torch
from rl_engine.reference import reduction as reference

expected = reference.reduce(values, combine_fn=torch.bitwise_xor, identity=0)
expected = reference.reduce(
    storage, length=n, from_memory=True, combine_fn=torch.bitwise_xor, identity=0,
)
merged = reference.reduce_partials(partials, combine_fn=torch.bitwise_xor, initial=0)
```

The callback is compiled into the enclosing Triton kernel, not invoked through
a runtime Python function pointer. Public custom-callback GPU tests cover loaded
prefixes, pointer tiling, parallel partials and merging, including a nonidentity
runtime seed. CPU-only runs do not qualify GPU compilation.

## Matching PyTorch reference

The reference package exports the same six names:

```python
import torch
from rl_engine.reference import reduction as reference

with torch.no_grad():
    # Models an already-loaded, power-of-two vector.
    loaded = reference.sum(values, length=128)

    # Models the scalar-pointer path, including fixed tiles and the outer seed.
    memory = reference.sum(storage, length=2500, from_memory=True)

    # Partial count comes from the final dimension.
    merged = reference.sum_partials(partials)
```

A Torch Tensor cannot distinguish a loaded JIT vector from the storage addressed
by a JIT pointer. The **reference-only** `from_memory=True` makes that distinction
explicit. Its default is False (loaded values); `length` never selects the mode.
Omitting `length` uses the complete last dimension in either mode. A provided
length cannot exceed that dimension. The loaded path masks without resizing;
the memory path selects the prefix before fixed-width tiling.

References accept CPU/GPU strided tensors, reduce the last dimension and retain
leading batch dimensions: `[M, V]` becomes `[M]`. `compute_dtype=None` keeps input
precision. Run eagerly and use `no_grad` for inputs requiring gradients; this
forward reference does not prescribe a Triton backward formula. PyTorch needs
no pipeline scheduling counterpart. Its optional `merge_unroll` mirrors the
partial fold's unrolled groups and tail.

Keep native PyTorch or higher-precision numerical-accuracy checks separate.
A same-tree reference is not an independent mathematical oracle or performance
baseline. Precision support in PyTorch does not establish support in a backend.

## Optional arithmetic configuration

Defaults come from `contracts/operators/ordered_reduction.py`:

| Triton parameter | PyTorch parameter | Default | Meaning |
| --- | --- | ---: | --- |
| `COMPUTE_DTYPE` | `compute_dtype` | `None` | Input dtype, unless explicitly overridden |
| `BLOCK_SIZE` | `block_size` | 1024 | Pointer/memory path's arithmetic tile width only |
| `MAX_GROUP_SIZE` | `max_group_size` | 8 | Maximum adjacent values folded sequentially inside each tile |
| `SECTION_COUNT` | `section_count` | 4 | Maximum sections used to combine group totals |

All arithmetic sizes must be positive powers of two. The loaded path uses its
vector width instead of `BLOCK_SIZE`. For matching nondefault pointer references:

```python
# Inside @triton.jit:
total = ordered.sum(
    ptr, length=V, COMPUTE_DTYPE=tl.float32,
    BLOCK_SIZE=2048, MAX_GROUP_SIZE=16, SECTION_COUNT=8,
)

# In eager PyTorch:
expected = reference.sum(
    x, length=V, from_memory=True, compute_dtype=torch.float32,
    block_size=2048, max_group_size=16, section_count=8,
)
```

Widths, group/section sizes and cast points can change rounding. Keep them fixed
across paths requiring identical bits. `COMPUTE_DTYPE` controls input conversion
and arithmetic, not just the output cast. No automatic widening is applied;
choose an integer accumulator to count booleans. Backend-supported types are
accepted without a utility-specific whitelist. Fixed order does not prevent
rounding, overflow or underflow, or promise reproducibility across hardware and
compiler versions. FP64 cannot restore input bits already lost.

## Effective-prefix invariance

For pointer input, preserve the effective length, arithmetic tile width and
precision while changing storage/batching:

```python
@triton.jit
def prefix_sum_kernel(x_ptr, out_ptr, row_stride, VALID_LENGTH: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    total = ordered.sum(
        x_ptr + row * row_stride, length=VALID_LENGTH, COMPUTE_DTYPE=tl.float32,
    )
    tl.store(out_ptr + row, total)
```

A 1000-value prefix uses `length=1000` whether allocated row width is 1000 or
2000. With `BLOCK_SIZE=1024`, its last 24 arithmetic lanes are padded with zero
without loading physical suffix values. The corresponding reference is
`reference.sum(storage, length=1000, from_memory=True, compute_dtype=torch.float32)`.
Tile boundaries are relative to the logical row start; callers own row addressing.

For loaded values, preserve the **loaded vector width** as well as length and
tree parameters. Masking a prefix does not make different loaded widths share
a tree. A caller requiring storage-width independence can use the pointer path
or consistently load the same arithmetic tiles itself.

This is not an incremental-prefix contract: a new valid element can change its
tile's partial; `reduce(x) + new_value` need not equal the recomputed longer sum.
`length` and `n_tiles` remain compile-time parameters. Different lengths can use
cached specializations; per-row lengths loaded from device memory are not yet
supported. Strided/paged pointer gathers remain caller-owned.

## Fused and parallel integration

Use `sum(values)` or `product(values)` after computing and masking one fixed-width
tile. Separate programs can store unseeded tile results in canonical slots and
then use the partial merge helpers. Serial and parallel paths must keep the
same tile boundaries, group/section sizes, dtype and outer seed. Convenience and generic
exports cover loaded values, complete memory vectors and stored partials.

<details markdown="1">
<summary>Internal stages and conformance checks</summary>

### Files and responsibilities

| File | Responsibility |
| --- | --- |
| `contracts/operators/ordered_reduction.py` | Shared arithmetic defaults; no Torch or Triton dependency |
| `backends/shared/triton/reduction/ordered_reduce.py` | JIT helpers embedded in a caller's kernel |
| `reference/reduction/ordered_reduce.py` | Eager PyTorch translation, with matching names, order and arithmetic |
| Each package's `__init__.py` | Convenience and generic helpers through one entry point; the only `__all__` list |
| `tests/helpers/ordered_reduce_kernels.py` | Test-only launch adapters; no production operator dependencies |

Existing reductions retain their own contracts: `reduce_rows_fp32` in
`rl_engine/ops/autograd/vjp_fp32.py` folds dimension zero for parameter gradients;
Attention and GEMM have specialized trees; distributed collectives live in
`rl_engine/distributed/algorithms/collectives.py`. Replacing any of these with a
different tree or axis is not a bitwise-equivalent substitution.

### Internal helpers

| Helper | Input and result |
| --- | --- |
| `reduce_add` / `reduce_multiply` | Pure pairwise `+` / `*` combiners |
| `ordered_tile_reduce` | One loaded power-of-two vector → scalar; no seed or dtype cast |
| `ordered_reduce_partial` | Vector pointer and tile index → one unseeded partial, with conversion and neutral tail padding |
| `ordered_reduce_serial` | Full vector → scalar; fold tiles from `initial` within one program |
| `ordered_reduce_merge` | Stored partials → scalar; fold ascending tile slots from `initial` |

Triton's configurable `ordered_*` helpers require all their arithmetic
arguments. Public `sum`/`product` share the public `reduce` implementation,
which selects the input representation, resolves precision, and applies a
loaded-prefix mask if needed. Custom callers use this same implementation.
Loaded values go to `ordered_tile_reduce`; scalar pointers go to
`ordered_reduce_serial`. This dispatch adds no new arithmetic tree.
The public `sum_partials`/`product_partials` helpers supply addition/zero or
multiplication/one to `reduce_partials`. That generic public entry resolves
storage precision and calls `ordered_reduce_merge`. These wrappers define
usable defaults; the generic helpers remain the single arithmetic implementation.
The PyTorch counterparts use lowercase parameter
names; `from_memory=True` models a pointer, and omitted lengths come from tensors.
Leading dimensions batch independent vectors. PyTorch omits the scheduling-only `PIPELINE_STAGES` option.

The serial stage calls `ordered_reduce_partial` to load, cast and pad each tile,
then `ordered_tile_reduce` determines its combine pairs. Separate producers can
use the same partial stage, and `ordered_reduce_merge` folds their stored results.
Each stage has a distinct role; the small `reduce_add`/`reduce_multiply` functions
are the binary callbacks required by the shared tree and fold.

The PyTorch-only `ordered_reduce` collector composes partial and merge helpers,
returning `OrderedReduceResult.partials` and `.total` for stage diagnostics.
Its last dimension must be nonempty. It does not implement another tree.

### Fixed tile tree and partial fold

1. Split a tile into adjacent groups of `min(BLOCK_SIZE, MAX_GROUP_SIZE)` values.
   Fold each group left to right, starting from its first value.
2. Split group totals into `min(N_GROUPS, SECTION_COUNT)` equal sections.
   Repeatedly pair each section's lower and upper halves until one total remains.
3. Merge section totals using the same halving-distance pairs. Four sections
   produce `(s0 + s2) + (s1 + s3)` for addition.

Each tree step explicitly selects its lower/upper operands and calls the binary
combiner. Triton uses `tl.reduce` only on an axis of exactly two entries, with
no implicit dtype widening. Built-in and caller-owned JIT combiners share this
path. A compile-time Python namespace exposes the original function to Triton
3.7.1 without turning it into a bound method; it adds no GPU operation.
PyTorch explicitly selects the same operand slices. The two-entry lowering cannot
choose a different whole-section tree because each level remains explicit.
A one-entry stage passes through without adding an identity. Neither delegates
the complete tile or section tree to an unconstrained native reduction.

The callback adapter targets Triton 3.7.1. Compile-time tree sizes are converted
to Python `int` before `bit_length()` so interpreter-mode constexpr values use
the same loop bounds. The bounded interpreter test checks the tree and public
entry points in a fresh process, with a 30-second timeout.

The standalone public API, prefix/batch/padding matrix and performance still
require target-device qualification. CPU reference tests do not qualify GPU
compilation, and CUDA evidence does not establish ROCm support.

Across tiles, serial and parallel schedules both compute
`(((initial + partial[0]) + partial[1]) + ...)` for addition. This is a left
fold, not another tile tree. `MERGE_UNROLL` accepts any positive integer and
expands loop steps without changing the pairs; incomplete tails preserve order.
Zero partials return `initial`. **Do not use `sum` to merge partials**:
that would introduce another tile tree and change the arithmetic.

`IDENTITY` and `initial` have distinct purposes. The identity fills padded lanes
and must be neutral for the combiner; `initial` enters the whole-vector fold
exactly once. For `10 + sum(x)`, use padding zero and initial ten. Partials must
not contain the seed. Seeds may be runtime scalars; stored partials must already
have the requested compute dtype.

Custom combiners must preserve operand shape and dtype and have a valid padding
identity. These helpers target mathematically associative, commutative operations;
floating-point rounding still depends on the tree. Multi-field statistics and
arbitrary pairwise subtraction or division require a different formulation.
The built-in combiners simply apply `+` or `*`; explicit per-operation casts,
NaN rules and signed-zero behavior belong to the chosen combiner.

### Fused and parallel callers

A fused caller uses the loaded-value calls shown above. Apply neutral padding
**after** the transformation, then use the returned scalar in further computation.

For explicit cross-program parallelism, the operator allocates partial storage,
launches producers into fixed tile slots, then launches a dependent merge.
Use the same stream or an explicit dependency across streams. JIT helpers cannot
synchronize independent programs. Partial storage and compute dtypes must match
so the parallel path introduces no extra narrowing.

Keep compiler settings consistent across paths. For composed arithmetic,
`enable_fp_fusion=False` preserves separate multiply/add rounding; denormal and
elementary-function settings remain the enclosing operator's responsibility.
A long serial merge may be expensive. Replacing it with another arithmetic tree
requires coordinated updates to every matching strategy and reference.

### Tests

```bash
python -m pytest tests/reference/reduction tests/backends/shared/triton/reduction -q
```

Coverage includes public XOR/modular-add callbacks with independent scalar oracles,
type-based public dispatch, loaded-prefix masks with poisoned suffixes, invalid
lengths, retained loaded width, frozen nonassociative examples, the original 1024/8/4 tree,
small and large tiles, alternative group/section sizes, strided reference inputs,
tails, empty vectors, custom seeds, compute precision, and serial/pipelined/parallel
conformance. Dedicated prefix tests compare the same values across batch sizes,
row positions and physical strides with NaN-poisoned suffixes. The GPU matrix
also compares serial, three-stage pipelined and parallel schedules, including
reversed producer-to-tile assignment. These are bitwise conformance checks;
independent exact-math and FP64 accuracy tests remain separate.

Run the focused prefix checks with:

```bash
python -m pytest tests/reference/reduction/test_prefix_invariance_reference.py tests/backends/shared/triton/reduction/test_prefix_invariance.py -q
```

Integer, boolean and floating cases exercise their respective
supported operations. Embedded tests check package imports, result use in further
computation, input preservation, output bounds and different warp counts.

Independent exact-math cases and native higher-precision comparisons supplement
fixed-path checks. Sharing constants must not silently redefine expected results.
GPU tests skip on CPU hosts; a CPU pass does not qualify Triton compilation,
GPU numerical behavior or performance.

The CPU CI job runs `tests/reference/reduction`. This includes the independent
accuracy checks as well as fixed-path conformance; neither replaces the other.

</details>
