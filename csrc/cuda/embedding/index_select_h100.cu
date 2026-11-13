// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// H100 index_select: copies in forward, one ordered FP32 chain per input
// coordinate in backward. Integer atomics assign position slots; those slots
// are sorted before any gradient is read. Floating-point atomics and partial
// sums would change the public rounding contract and are deliberately absent.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <ATen/cuda/detail/IntegerDivider.cuh>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxRank = 32;
constexpr int64_t kWorkspaceBytes = 128LL << 20;
constexpr unsigned kWarpMask = 0xffffffffu;
using Divider = at::cuda::detail::IntDivider<unsigned int>;

// Shape and strides describe the original tensor, never a materialized copy.
// The logical [outer, axis, inner] view keeps every non-selected coordinate
// independent, including when the selected dimension is first or last.
struct Layout {
  int rank;
  int dim;
  int axis;
  int inner;
  int outer;
  Divider inner_divider;
  Divider input_axis_divider;
  Divider column_tile_divider;
  int64_t sizes[kMaxRank];
  int64_t strides[kMaxRank];

  __device__ __forceinline__ int64_t offset(int outer_index, int position, int column) const {
    int64_t result = static_cast<int64_t>(position) * strides[dim];
    for (int d = dim - 1; d >= 0; --d) {
      result += (outer_index % sizes[d]) * strides[d];
      outer_index /= sizes[d];
    }
    for (int d = rank - 1; d > dim; --d) {
      result += (column % sizes[d]) * strides[d];
      column /= sizes[d];
    }
    return result;
  }
};

Layout make_layout(const torch::Tensor& tensor, int dim) {
  Layout layout{};
  layout.rank = tensor.dim();
  layout.dim = dim;
  layout.axis = tensor.size(dim);
  layout.outer = layout.inner = 1;
  for (int d = 0; d < layout.rank; ++d) {
    layout.sizes[d] = tensor.size(d);
    layout.strides[d] = tensor.stride(d);
    if (d < dim) {
      TORCH_CHECK(static_cast<int64_t>(layout.outer) * tensor.size(d) < INT_MAX,
                  "H100 outer size must fit int32");
      layout.outer *= tensor.size(d);
    }
    if (d > dim) {
      TORCH_CHECK(static_cast<int64_t>(layout.inner) * tensor.size(d) < INT_MAX,
                  "H100 inner size must fit int32");
      layout.inner *= tensor.size(d);
    }
  }
  layout.inner_divider = Divider(std::max(layout.inner, 1));
  return layout;
}

void check_device(const torch::Tensor& tensor) {
  TORCH_CHECK(tensor.is_cuda(), "index_select requires NVIDIA CUDA");
  const auto* properties = at::cuda::getDeviceProperties(tensor.get_device());
  TORCH_CHECK(properties->major == 9 && properties->minor == 0 &&
                  std::strcmp(properties->name, "NVIDIA H100 NVL") == 0,
              "index_select_h100 is qualified on NVIDIA H100 NVL only");
}

int check_inputs(const torch::Tensor& x, const torch::Tensor& index, int64_t dim) {
  check_device(x);
  TORCH_CHECK_VALUE(x.dim() > 0 && x.dim() <= kMaxRank, "x must have between 1 and 32 dimensions");
  TORCH_CHECK_INDEX(dim >= -x.dim() && dim < x.dim(), "dim must be in [-rank, rank)");
  TORCH_CHECK_TYPE(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
                       x.scalar_type() == at::kBFloat16,
                   "x dtype must be float32, float16, or bfloat16");
  TORCH_CHECK_VALUE(index.dim() == 1, "index must have shape [K]");
  TORCH_CHECK_TYPE(index.scalar_type() == at::kInt || index.scalar_type() == at::kLong,
                   "index dtype must be int32 or int64");
  TORCH_CHECK_VALUE(x.device() == index.device(), "x and index must be on the same device");
  TORCH_CHECK(x.numel() < INT_MAX && index.numel() < INT_MAX,
              "H100 specialization requires fewer than INT_MAX elements");
  for (int d = 0; d < x.dim(); ++d) {
    TORCH_CHECK(x.size(d) < INT_MAX, "H100 specialization requires int32 axis sizes");
  }
  return dim < 0 ? dim + x.dim() : dim;
}

template <typename index_t>
__global__ void check_bounds(const index_t* index, int count, int64_t stride, int axis) {
  for (int64_t p = blockIdx.x * blockDim.x + threadIdx.x; p < count; p += blockDim.x * gridDim.x) {
    const int64_t source = index[static_cast<int64_t>(p) * stride];
    CUDA_KERNEL_ASSERT(source >= 0 && source < axis);
  }
}

template <typename value_t, typename index_t, bool Contiguous>
__global__ void copy_selected(const value_t* x, const index_t* index, value_t* y, Layout layout,
                              Divider selected_count_divider, int elements, int64_t index_stride) {
  const int inner = layout.inner;
  for (int64_t q = blockIdx.x * blockDim.x + threadIdx.x; q < elements;
       q += blockDim.x * gridDim.x) {
    const auto coordinate = layout.inner_divider.divmod(static_cast<unsigned int>(q));
    const int column = coordinate.mod;
    const auto selected = selected_count_divider.divmod(coordinate.div);
    const int position = selected.mod;
    const int outer = selected.div;
    const int64_t source = index[static_cast<int64_t>(position) * index_stride];
    const bool valid = source >= 0 && source < layout.axis;
    CUDA_KERNEL_ASSERT(valid);
    if (valid) {
      const int64_t offset =
          Contiguous ? (static_cast<int64_t>(outer) * layout.axis + source) * inner + column
                     : layout.offset(outer, source, column);
      y[q] = x[offset];
    }
  }
}

template <typename index_t, int Slots>
__global__ void group_positions(const index_t* index, int* counts, int* positions,
                                int* queued_sources, int count, int axis, int64_t index_stride,
                                bool row_major) {
  for (int64_t p = blockIdx.x * blockDim.x + threadIdx.x; p < count; p += blockDim.x * gridDim.x) {
    const int64_t source = index[static_cast<int64_t>(p) * index_stride];
    CUDA_KERNEL_ASSERT(source >= 0 && source < axis);
    if (source >= 0 && source < axis) {
      const int slot = atomicAdd(counts + source, 1);
      if (slot < Slots) {
        const int64_t offset =
            row_major ? source * Slots + slot : static_cast<int64_t>(slot) * axis + source;
        positions[offset] = p;
      }
      // Exactly one insertion per group with more than four occurrences.
      if (slot == 4) queued_sources[atomicAdd(counts + axis, 1)] = source;
    }
  }
}

__device__ __forceinline__ void compare_swap(int& a, int& b) {
  const int low = min(a, b);
  b = max(a, b);
  a = low;
}

// A warp owns Slots integer positions in striped order. Sorting indices is
// independent of the FP32 chain; the consumer broadcasts them in increasing p.
template <int Slots>
__device__ __forceinline__ void sort_warp_positions(int (&values)[Slots / 32]) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int size = 2; size <= Slots; size *= 2) {
#pragma unroll
    for (int step = size / 2; step; step /= 2) {
      int other[Slots / 32];
#pragma unroll
      for (int item = 0; item < Slots / 32; ++item) {
        if (step < 32) {
          other[item] = __shfl_xor_sync(kWarpMask, values[item], step);
        } else {
          other[item] = values[item ^ (step / 32)];
        }
      }
#pragma unroll
      for (int item = 0; item < Slots / 32; ++item) {
        const int position = lane + 32 * item;
        const bool take_min = ((position & size) == 0) == ((position & step) == 0);
        values[item] = take_min ? min(values[item], other[item]) : max(values[item], other[item]);
      }
    }
  }
}

template <typename value_t, bool Contiguous>
__device__ __forceinline__ float read_grad(const value_t* grad, const Layout& layout, int outer,
                                           int position, int column) {
  const int64_t offset =
      Contiguous ? (static_cast<int64_t>(outer) * layout.axis + position) * layout.inner + column
                 : layout.offset(outer, position, column);
  return static_cast<float>(grad[offset]);
}

// Four columns per lane amortize metadata loads for widths 32/96/128 and wider.
// Small groups use only four sorted scalar positions. Larger groups sort bounded
// integer slots across a warp. An overflowing group scans index in 32-position
// windows; ballots enumerate matches in original order without a linked-list
// dependency or an unbounded scratch allocation.
template <typename value_t, typename index_t, bool Contiguous, int Slots, bool Scan,
          bool Queued = false>
__global__ void ordered_rows(const value_t* grad, const index_t* index, const int* counts,
                             const int* positions, value_t* result, Layout layout, int input_axis,
                             int64_t index_stride, int column_tiles, int64_t work_items,
                             const int* queued_sources) {
  const int lane = threadIdx.x & 31;
  const int first_warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int queue_count = Queued ? counts[input_axis] : 0;
  const int64_t active_work =
      Queued ? static_cast<int64_t>(queue_count) * layout.outer * column_tiles : work_items;
  for (int64_t work = first_warp; work < active_work; work += gridDim.x * (blockDim.x / 32)) {
    const auto tiled = layout.column_tile_divider.divmod(static_cast<unsigned int>(work));
    const int tile = tiled.mod;
    const auto dense = layout.input_axis_divider.divmod(tiled.div);
    const int source = Queued ? queued_sources[tiled.div % queue_count] : dense.mod;
    const int outer = Queued ? tiled.div / queue_count : dense.div;
    const int64_t row = static_cast<int64_t>(outer) * input_axis + source;
    const int first_column = tile * 128 + lane;
    float total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int occurrences = Scan ? layout.axis : counts[source];
    if (!Scan && !Queued && occurrences <= 4) {
      int p[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        p[i] = i < occurrences ? positions[static_cast<int64_t>(source) * Slots + i] : INT_MAX;
      }
      compare_swap(p[0], p[1]);
      compare_swap(p[2], p[3]);
      compare_swap(p[0], p[2]);
      compare_swap(p[1], p[3]);
      compare_swap(p[1], p[2]);
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int column = first_column + item * 32;
        if (column < layout.inner) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            if (i < occurrences) {
              total[item] = __fadd_rn(
                  total[item], read_grad<value_t, Contiguous>(grad, layout, outer, p[i], column));
            }
          }
        }
      }
    } else if (!Scan && occurrences <= Slots) {
      int ordered[Slots / 32];
#pragma unroll
      for (int item = 0; item < Slots / 32; ++item) {
        const int slot = item * 32 + lane;
        ordered[item] =
            slot < occurrences ? positions[static_cast<int64_t>(source) * Slots + slot] : INT_MAX;
      }
      sort_warp_positions<Slots>(ordered);
      for (int start = 0; start < occurrences; start += 4) {
        int selected[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int p = start + i;
          int value = INT_MAX;
#pragma unroll
          for (int item = 0; item < Slots / 32; ++item) {
            if (p / 32 == item) value = ordered[item];
          }
          selected[i] = __shfl_sync(kWarpMask, value, p & 31);
        }
#pragma unroll
        for (int item = 0; item < 4; ++item) {
          const int column = first_column + item * 32;
          float values[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            values[i] =
                column < layout.inner && start + i < occurrences
                    ? read_grad<value_t, Contiguous>(grad, layout, outer, selected[i], column)
                    : 0.0f;
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            if (start + i < occurrences) total[item] = __fadd_rn(total[item], values[i]);
          }
        }
      }
    } else {
      // Prefetch independent index windows before consuming their matches.
      // Ballots only compact integer positions; gradient additions still run
      // through each window and each set bit in original position order.
      constexpr int windows = Scan ? 1 : 8;
      for (int64_t start = 0; start < layout.axis; start += 32 * windows) {
        unsigned match_bits = 0;
#pragma unroll
        for (int window = 0; window < windows; ++window) {
          const int64_t position = start + window * 32 + lane;
          const bool matches = position < layout.axis && index[position * index_stride] == source;
          match_bits |= static_cast<unsigned>(matches) << window;
        }
#pragma unroll 1
        for (int window = 0; window < windows; ++window) {
          unsigned remaining = __ballot_sync(kWarpMask, (match_bits >> window) & 1u);
          while (remaining) {
            const int p = start + window * 32 + __ffs(remaining) - 1;
#pragma unroll
            for (int item = 0; item < 4; ++item) {
              const int column = first_column + item * 32;
              if (column < layout.inner) {
                total[item] = __fadd_rn(
                    total[item], read_grad<value_t, Contiguous>(grad, layout, outer, p, column));
              }
            }
            remaining &= remaining - 1;
          }
        }
      }
    }
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int column = first_column + item * 32;
      if (column < layout.inner) {
        result[row * layout.inner + column] = static_cast<value_t>(total[item]);
      }
    }
  }
}

// Keep the common zero/one/few-occurrence path out of the sorting kernel's
// register allocation. This kernel and the queued kernel write disjoint rows.
template <typename value_t, bool Contiguous>
__global__ void ordered_short_rows(const value_t* grad, const int* counts, const int* positions,
                                   value_t* result, Layout layout, int input_axis, int slots,
                                   int64_t work_items) {
  const int lane = threadIdx.x & 31;
  const int first_warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  for (int64_t work = first_warp; work < work_items; work += gridDim.x * (blockDim.x / 32)) {
    const auto tiled = layout.column_tile_divider.divmod(static_cast<unsigned int>(work));
    const auto selected = layout.input_axis_divider.divmod(tiled.div);
    const int source = selected.mod;
    const int count = counts[source];
    if (count > 4) continue;
    int p[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      p[i] = i < count ? positions[static_cast<int64_t>(source) * slots + i] : INT_MAX;
    }
    compare_swap(p[0], p[1]);
    compare_swap(p[2], p[3]);
    compare_swap(p[0], p[2]);
    compare_swap(p[1], p[3]);
    compare_swap(p[1], p[2]);
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int column = tiled.mod * 128 + lane + item * 32;
      if (column < layout.inner) {
        float total = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if (i < count)
            total = __fadd_rn(
                total, read_grad<value_t, Contiguous>(grad, layout, selected.div, p[i], column));
        }
        result[static_cast<int64_t>(tiled.div) * layout.inner + column] =
            static_cast<value_t>(total);
      }
    }
  }
}

// A 16-byte pack keeps contiguous short groups on vector memory operations.
// Each lane still owns independent FP32 accumulators; packing does not combine
// values or change their addition order. The union is only a storage-bit view.
template <typename value_t>
union alignas(16) ValuePack {
  uint4 storage;
  value_t values[16 / sizeof(value_t)];
  __device__ ValuePack() {}
};

template <typename value_t>
__global__ void ordered_short_vectors(const value_t* grad, const int* counts, const int* positions,
                                      value_t* result, Layout layout, Divider vector_inner_divider,
                                      int slots, int vectors) {
  constexpr int width = 16 / sizeof(value_t);
  for (int64_t q = blockIdx.x * blockDim.x + threadIdx.x; q < vectors;
       q += blockDim.x * gridDim.x) {
    const auto row = vector_inner_divider.divmod(static_cast<unsigned int>(q));
    const auto selected = layout.input_axis_divider.divmod(row.div);
    const int count = counts[selected.mod];
    if (count > 4) continue;
    int p[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      p[i] = i < count ? positions[static_cast<int64_t>(selected.mod) * slots + i] : INT_MAX;
    }
    compare_swap(p[0], p[1]);
    compare_swap(p[2], p[3]);
    compare_swap(p[0], p[2]);
    compare_swap(p[1], p[3]);
    compare_swap(p[1], p[2]);
    float total[width] = {};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (i < count) {
        const int64_t offset =
            (static_cast<int64_t>(selected.div) * layout.axis + p[i]) * layout.inner +
            row.mod * width;
        ValuePack<value_t> loaded;
        loaded.storage = *reinterpret_cast<const uint4*>(grad + offset);
#pragma unroll
        for (int element = 0; element < width; ++element) {
          total[element] = __fadd_rn(total[element], static_cast<float>(loaded.values[element]));
        }
      }
    }
    ValuePack<value_t> packed;
#pragma unroll
    for (int element = 0; element < width; ++element) {
      packed.values[element] = static_cast<value_t>(total[element]);
    }
    reinterpret_cast<uint4*>(result)[q] = packed.storage;
  }
}

// Last-axis / rank-one selection: consecutive threads own consecutive source
// coordinates. This avoids wasting 31 lanes on every scalar input row.
template <typename value_t, typename index_t, bool Contiguous>
__global__ void ordered_scalars(const value_t* grad, const index_t* index, const int* counts,
                                const int* positions, value_t* result, Layout layout,
                                int input_axis, int64_t index_stride, int elements) {
  for (int64_t q = blockIdx.x * blockDim.x + threadIdx.x; q < elements;
       q += blockDim.x * gridDim.x) {
    const auto coordinate = layout.input_axis_divider.divmod(static_cast<unsigned int>(q));
    const int source = coordinate.mod;
    const int outer = coordinate.div;
    const int count = counts[source];
    float total = 0.0f;
    if (count <= 4) {
      int p[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        p[i] = i < count ? positions[static_cast<int64_t>(i) * input_axis + source] : INT_MAX;
      }
      compare_swap(p[0], p[1]);
      compare_swap(p[2], p[3]);
      compare_swap(p[0], p[2]);
      compare_swap(p[1], p[3]);
      compare_swap(p[1], p[2]);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i < count)
          total = __fadd_rn(total, read_grad<value_t, Contiguous>(grad, layout, outer, p[i], 0));
      }
    } else {
      for (int p = 0; p < layout.axis; ++p) {
        if (index[static_cast<int64_t>(p) * index_stride] == source) {
          total = __fadd_rn(total, read_grad<value_t, Contiguous>(grad, layout, outer, p, 0));
        }
      }
    }
    result[q] = static_cast<value_t>(total);
  }
}

int blocks(int64_t elements, int threads = kThreads) {
  return std::min<int64_t>((elements + threads - 1) / threads, 65535);
}

}  // namespace

torch::Tensor index_select_h100_forward(torch::Tensor x, torch::Tensor index, int64_t dim) {
  dim = check_inputs(x, index, dim);
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto shape = x.sizes().vec();
  shape[dim] = index.numel();
  auto output = torch::empty(shape, x.options());
  TORCH_CHECK(output.numel() < INT_MAX, "H100 output must have fewer than INT_MAX elements");
  if (index.numel() == 0) return output;
  const Layout layout = make_layout(x, dim);
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto index_tag) {
    using index_t = decltype(index_tag);
    const auto* indices = index.data_ptr<index_t>();
    if (output.numel() == 0) {
      check_bounds<<<blocks(index.numel()), kThreads, 0, stream>>>(indices, index.numel(),
                                                                   index.stride(0), layout.axis);
      return;
    }
    const int vector_elements = 16 / x.element_size();
    if (x.is_contiguous() && layout.inner % vector_elements == 0 &&
        reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0) {
      // Copy 16 storage bytes per lane. No conversion is involved, including
      // for FP16/BF16 NaN payloads and signed zero.
      Layout vector_layout = layout;
      vector_layout.inner /= vector_elements;
      vector_layout.inner_divider = Divider(vector_layout.inner);
      const int vectors = output.numel() / vector_elements;
      copy_selected<uint4, index_t, true><<<blocks(vectors), kThreads, 0, stream>>>(
          reinterpret_cast<const uint4*>(x.data_ptr()), indices,
          reinterpret_cast<uint4*>(output.data_ptr()), vector_layout, Divider(index.numel()),
          vectors, index.stride(0));
      return;
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(), "index_select_h100_forward", [&] {
          const auto* input = x.data_ptr<scalar_t>();
          auto* result = output.data_ptr<scalar_t>();
          const int grid = blocks(output.numel());
          if (x.is_contiguous()) {
            copy_selected<scalar_t, index_t, true><<<grid, kThreads, 0, stream>>>(
                input, indices, result, layout, Divider(index.numel()), output.numel(),
                index.stride(0));
          } else {
            copy_selected<scalar_t, index_t, false><<<grid, kThreads, 0, stream>>>(
                input, indices, result, layout, Divider(index.numel()), output.numel(),
                index.stride(0));
          }
        });
  };
  if (index.scalar_type() == at::kLong)
    launch(int64_t{0});
  else
    launch(int32_t{0});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor index_select_h100_backward(torch::Tensor grad, torch::Tensor index,
                                         std::vector<int64_t> input_shape, int64_t dim) {
  dim = check_inputs(grad, index, dim);
  TORCH_CHECK(input_shape.size() == static_cast<size_t>(grad.dim()), "input shape rank mismatch");
  TORCH_CHECK(grad.size(dim) == index.numel(), "gradient axis must have K entries");
  for (int d = 0; d < grad.dim(); ++d) {
    TORCH_CHECK(input_shape[d] >= 0 && input_shape[d] < INT_MAX, "invalid input shape");
    TORCH_CHECK(d == dim || input_shape[d] == grad.size(d), "gradient shape mismatch");
  }
  const c10::cuda::CUDAGuard device_guard(grad.device());
  if (grad.numel() == 0) return torch::zeros(input_shape, grad.options());
  auto result = torch::empty(input_shape, grad.options());
  TORCH_CHECK(result.numel() < INT_MAX, "H100 gradient must have fewer than INT_MAX elements");
  Layout layout = make_layout(grad, dim);
  const int axis = input_shape[dim];
  TORCH_CHECK(axis > 0, "nonempty selection requires a nonempty input axis");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const bool scan = index.numel() <= 64 && axis <= 256 && layout.outer <= 16;
  const int tiles = (static_cast<int64_t>(layout.inner) + 127) / 128;
  layout.input_axis_divider = Divider(axis);
  layout.column_tile_divider = Divider(tiles);
  const int64_t queue_capacity = std::max<int64_t>(1, index.numel() / 5);
  const bool full_slots_fit =
      (static_cast<int64_t>(axis) * 129 + 1 + queue_capacity) * 4 <= kWorkspaceBytes;
  const int slots = layout.inner == 1 ? 4 : full_slots_fit ? 128 : 32;
  const int64_t metadata_elements = static_cast<int64_t>(axis) * (slots + 1) + 1 + queue_capacity;
  TORCH_CHECK(scan || metadata_elements * 4 <= kWorkspaceBytes,
              "H100 integer workspace would exceed 128 MiB");
  torch::Tensor metadata;
  int* counts = nullptr;
  int* positions = nullptr;
  int* queued_sources = nullptr;
  if (!scan) {
    metadata = torch::empty({metadata_elements}, index.options().dtype(at::kInt));
    counts = metadata.data_ptr<int>();
    positions = counts + axis + 1;
    queued_sources = positions + static_cast<int64_t>(axis) * slots;
    C10_CUDA_CHECK(cudaMemsetAsync(counts, 0, (static_cast<size_t>(axis) + 1) * 4, stream));
  }
  auto launch = [&](auto index_tag) {
    using index_t = decltype(index_tag);
    const auto* indices = index.data_ptr<index_t>();
    if (!scan) {
      if (slots == 128) {
        group_positions<index_t, 128><<<blocks(index.numel()), kThreads, 0, stream>>>(
            indices, counts, positions, queued_sources, index.numel(), axis, index.stride(0), true);
      } else if (slots == 32) {
        group_positions<index_t, 32><<<blocks(index.numel()), kThreads, 0, stream>>>(
            indices, counts, positions, queued_sources, index.numel(), axis, index.stride(0), true);
      } else {
        group_positions<index_t, 4><<<blocks(index.numel()), kThreads, 0, stream>>>(
            indices, counts, positions, queued_sources, index.numel(), axis, index.stride(0),
            false);
      }
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, grad.scalar_type(), "index_select_h100_backward", [&] {
          const auto* input = grad.data_ptr<scalar_t>();
          auto* output = result.data_ptr<scalar_t>();
          auto launch_layout = [&](auto contiguous_tag) {
            constexpr bool contiguous = decltype(contiguous_tag)::value;
            if (!scan && layout.inner == 1) {
              ordered_scalars<scalar_t, index_t, contiguous>
                  <<<blocks(result.numel()), kThreads, 0, stream>>>(
                      input, indices, counts, positions, output, layout, axis, index.stride(0),
                      result.numel());
              return;
            }
            const int64_t work = static_cast<int64_t>(layout.outer) * axis * tiles;
            const int grid = blocks(work, kThreads / 32);
            const bool split_groups = !scan && result.numel() >= (1 << 18);
            if (split_groups) {
              constexpr int vector_width = 16 / sizeof(scalar_t);
              if (contiguous && layout.inner % vector_width == 0 &&
                  reinterpret_cast<uintptr_t>(input) % 16 == 0) {
                const int vectors = result.numel() / vector_width;
                ordered_short_vectors<scalar_t><<<blocks(vectors), kThreads, 0, stream>>>(
                    input, counts, positions, output, layout, Divider(layout.inner / vector_width),
                    slots, vectors);
              } else {
                ordered_short_rows<scalar_t, contiguous><<<grid, kThreads, 0, stream>>>(
                    input, counts, positions, output, layout, axis, slots, work);
              }
              const int queue_grid = blocks(
                  std::min<int64_t>(queue_capacity, 1024) * layout.outer * tiles, kThreads / 32);
              if (slots == 128) {
                ordered_rows<scalar_t, index_t, contiguous, 128, false, true>
                    <<<queue_grid, kThreads, 0, stream>>>(input, indices, counts, positions, output,
                                                          layout, axis, index.stride(0), tiles,
                                                          work, queued_sources);
              } else {
                ordered_rows<scalar_t, index_t, contiguous, 32, false, true>
                    <<<queue_grid, kThreads, 0, stream>>>(input, indices, counts, positions, output,
                                                          layout, axis, index.stride(0), tiles,
                                                          work, queued_sources);
              }
              return;
            }
            if (scan) {
              ordered_rows<scalar_t, index_t, contiguous, 32, true><<<grid, kThreads, 0, stream>>>(
                  input, indices, counts, positions, output, layout, axis, index.stride(0), tiles,
                  work, queued_sources);
            } else if (slots == 128) {
              ordered_rows<scalar_t, index_t, contiguous, 128, false>
                  <<<grid, kThreads, 0, stream>>>(input, indices, counts, positions, output, layout,
                                                  axis, index.stride(0), tiles, work,
                                                  queued_sources);
            } else {
              ordered_rows<scalar_t, index_t, contiguous, 32, false><<<grid, kThreads, 0, stream>>>(
                  input, indices, counts, positions, output, layout, axis, index.stride(0), tiles,
                  work, queued_sources);
            }
          };
          if (grad.is_contiguous())
            launch_layout(std::true_type{});
          else
            launch_layout(std::false_type{});
        });
  };
  if (index.scalar_type() == at::kLong)
    launch(int64_t{0});
  else
    launch(int32_t{0});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}
