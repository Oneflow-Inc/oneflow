/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_PRIMITIVE_COMMON_PERMUTE_H_
#define ONEFLOW_CORE_PRIMITIVE_COMMON_PERMUTE_H_

#include "oneflow/core/primitive/include/primitive.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace primitive {

namespace permute_internal {

namespace {

template<size_t max_movement_size>
size_t GetMovementSize(size_t elem_size, size_t num_dims, const int64_t* src_dims, const void* src,
                       const int* permutation, void* dst) {
  static_assert(max_movement_size > 0 && (max_movement_size & (max_movement_size - 1)) == 0, "");
  CHECK_GT(elem_size, 0);
  CHECK_EQ((elem_size & (elem_size - 1)), 0);
  CHECK_EQ(max_movement_size % elem_size, 0);
  if (permutation[num_dims - 1] == num_dims - 1) {
    const int64_t last_dim_size = src_dims[num_dims - 1] * elem_size;
    auto src_ptr = reinterpret_cast<std::uintptr_t>(src);
    auto dst_ptr = reinterpret_cast<std::uintptr_t>(dst);
    for (size_t size = max_movement_size; size > elem_size; size /= 2) {
      if (last_dim_size % size == 0 && src_ptr % size == 0 && dst_ptr % size == 0) { return size; }
    }
  }
  return elem_size;
}

template<size_t max_num_dims>
void SimplifyPermutation(size_t num_dims, const int64_t* src_dims, const int* permutation,
                         size_t* simplified_num_dims, int64_t* simplified_src_dims,
                         int* simplified_permutation) {
  CHECK_NE(num_dims, 0);
  int64_t coalesced_dims[max_num_dims];
  size_t start_permutation_index = 0;
  while (start_permutation_index < num_dims) {
    const size_t start_dim_index = permutation[start_permutation_index];
    coalesced_dims[start_dim_index] = src_dims[start_dim_index];
    size_t end_permutation_index = start_permutation_index + 1;
    const size_t end_dim_index = permutation[end_permutation_index];
    while (end_permutation_index < num_dims
           && end_dim_index == permutation[end_permutation_index - 1] + 1) {
      coalesced_dims[start_dim_index] *= src_dims[end_dim_index];
      coalesced_dims[end_dim_index] = 1;
      end_permutation_index += 1;
    }
    start_permutation_index = end_permutation_index;
  }
  size_t valid_num_dims = 0;
  int mapping[max_num_dims];
  for (size_t i = 0; i < num_dims; ++i) {
    const int src_dim = coalesced_dims[i];
    if (src_dim == 1) {
      mapping[i] = -1;
    } else {
      mapping[i] = valid_num_dims;
      simplified_src_dims[valid_num_dims] = src_dim;
      valid_num_dims += 1;
    }
  }
  if (valid_num_dims == 0) {
    *simplified_num_dims = 1;
    simplified_src_dims[0] = 1;
    simplified_permutation[0] = 0;
  } else {
    *simplified_num_dims = valid_num_dims;
    size_t permutation_index = 0;
    for (size_t i = 0; i < num_dims; ++i) {
      const int mapped = mapping[permutation[i]];
      if (mapped >= 0) {
        simplified_permutation[permutation_index] = mapped;
        permutation_index += 1;
      }
    }
  }
}

template<size_t max_num_dims, size_t max_movement_size>
void SimplifyPermutation(size_t num_dims, const int64_t* src_dims, const int* permutation,
                         size_t* simplified_num_dims, int64_t* simplified_src_dims,
                         int* simplified_permutation, size_t elem_size, const void* src, void* dst,
                         size_t* movement_size) {
  const size_t pre_simplified_movement_size =
      GetMovementSize<max_movement_size>(elem_size, num_dims, src_dims, src, permutation, dst);
  int64_t tmp_dims[max_num_dims];
  for (size_t i = 0; i < num_dims; ++i) { tmp_dims[i] = src_dims[i]; }
  tmp_dims[num_dims - 1] /= (pre_simplified_movement_size / elem_size);
  SimplifyPermutation<max_num_dims>(num_dims, tmp_dims, permutation, simplified_num_dims,
                                    simplified_src_dims, simplified_permutation);
  *movement_size =
      GetMovementSize<max_movement_size>(pre_simplified_movement_size, *simplified_num_dims,
                                         simplified_src_dims, src, simplified_permutation, dst);
  simplified_src_dims[*simplified_num_dims - 1] /= (*movement_size / pre_simplified_movement_size);
}

template<size_t num_dims, typename IndexType>
struct PermuteKernelParams {
  NdIndexOffsetHelper<IndexType, num_dims> src_index_helper;
  NdIndexOffsetHelper<IndexType, num_dims> dst_index_helper;
  int permutation[num_dims]{};
  IndexType count{};
  const void* src{};
  void* dst{};
};

constexpr size_t kMaxMovementSize = 16;
constexpr size_t kMaxNumDims = 8;

template<size_t num_dims, size_t movement_size, typename IndexType>
void LaunchKernel(StreamContext* stream_ctx, PermuteKernelParams<num_dims, IndexType> params);

template<size_t num_dims, size_t movement_size, typename IndexType>
void LaunchKernel(StreamContext* stream_ctx, const int64_t* src_dims, const void* src,
                  const int* permutation, void* dst, size_t count) {
  PermuteKernelParams<num_dims, IndexType> params;
  params.src_index_helper = NdIndexOffsetHelper<IndexType, num_dims>(src_dims);
  int64_t dst_dims[num_dims];
  for (size_t i = 0; i < num_dims; ++i) { dst_dims[i] = src_dims[permutation[i]]; }
  params.dst_index_helper = NdIndexOffsetHelper<IndexType, num_dims>(dst_dims);
  for (size_t i = 0; i < num_dims; ++i) { params.permutation[i] = permutation[i]; }
  params.src = src;
  params.dst = dst;
  params.count = static_cast<IndexType>(count);
  LaunchKernel<num_dims, movement_size, IndexType>(stream_ctx, params);
}

template<size_t num_dims, size_t movement_size>
void DispatchIndexType(StreamContext* stream_ctx, const int64_t* src_dims, const void* src,
                       const int* permutation, void* dst) {
  size_t count = 1;
  for (size_t i = 0; i < num_dims; ++i) { count *= src_dims[i]; }
  if (count < GetMaxVal<int32_t>()) {
    LaunchKernel<num_dims, movement_size, int32_t>(stream_ctx, src_dims, src, permutation, dst,
                                                   count);
  } else {
    LaunchKernel<num_dims, movement_size, int64_t>(stream_ctx, src_dims, src, permutation, dst,
                                                   count);
  }
}

template<size_t num_dims>
void DispatchMovementSize(StreamContext* stream_ctx, size_t movement_size, const int64_t* src_dims,
                          const void* src, const int* permutation, void* dst) {
  void (*func)(StreamContext* /*stream_ctx*/, const int64_t* /*src_dims*/, const void* /*src*/,
               const int* /*permutation*/, void* /*dst*/) = nullptr;
  if (movement_size == 1) {
    func = DispatchIndexType<num_dims, 1>;
  } else if (movement_size == 2) {
    func = DispatchIndexType<num_dims, 2>;
  } else if (movement_size == 4) {
    func = DispatchIndexType<num_dims, 4>;
  } else if (movement_size == 8) {
    func = DispatchIndexType<num_dims, 8>;
  } else if (movement_size == 16) {
    func = DispatchIndexType<num_dims, 16>;
  } else {
    UNIMPLEMENTED();
  }
  func(stream_ctx, src_dims, src, permutation, dst);
}

void LaunchWithSimplified(StreamContext* stream_ctx, size_t movement_size, size_t num_dims,
                          const int64_t* src_dims, const void* src, const int* permutation,
                          void* dst) {
  void (*func)(StreamContext* /*stream_ctx*/, size_t /*movement_size*/, const int64_t* /*src_dims*/,
               const void* /*src*/, const int* /*permutation*/, void* /*dst*/) = nullptr;
  if (num_dims == 1) {
    func = DispatchMovementSize<1>;
  } else if (num_dims == 2) {
    func = DispatchMovementSize<2>;
  } else if (num_dims == 3) {
    func = DispatchMovementSize<3>;
  } else if (num_dims == 4) {
    func = DispatchMovementSize<4>;
  } else if (num_dims == 5) {
    func = DispatchMovementSize<5>;
  } else if (num_dims == 6) {
    func = DispatchMovementSize<6>;
  } else if (num_dims == 7) {
    func = DispatchMovementSize<7>;
  } else if (num_dims == 8) {
    func = DispatchMovementSize<8>;
  } else {
    UNIMPLEMENTED();
  }
  func(stream_ctx, movement_size, src_dims, src, permutation, dst);
}

void SimplifyThenLaunch(StreamContext* stream_ctx, DataType data_type, size_t num_dims,
                        const int64_t* src_dims, const void* src, const int* permutation,
                        void* dst) {
  CHECK_LE(num_dims, kMaxNumDims);
  CHECK_GT(num_dims, 0);
  size_t simplified_num_dims = 0;
  int64_t simplified_src_dims[kMaxNumDims];
  int simplified_permutation[kMaxNumDims];
  size_t movement_size = 0;
  SimplifyPermutation<kMaxNumDims, kMaxMovementSize>(
      num_dims, src_dims, permutation, &simplified_num_dims, simplified_src_dims,
      simplified_permutation, GetSizeOfDataType(data_type), src, dst, &movement_size);
  LaunchWithSimplified(stream_ctx, movement_size, simplified_num_dims, simplified_src_dims, src,
                       simplified_permutation, dst);
}

}  // namespace

}  // namespace permute_internal

}  // namespace primitive

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PRIMITIVE_COMMON_PERMUTE_H_
