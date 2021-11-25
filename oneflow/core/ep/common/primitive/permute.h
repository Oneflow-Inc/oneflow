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
#ifndef ONEFLOW_CORE_EP_COMMON_PRIMITIVE_PERMUTE_H_
#define ONEFLOW_CORE_EP_COMMON_PRIMITIVE_PERMUTE_H_

#include "oneflow/core/ep/include/primitive/primitive.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace permute {

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
    while (end_permutation_index < num_dims
           && permutation[end_permutation_index] == permutation[end_permutation_index - 1] + 1) {
      const size_t end_dim_index = permutation[end_permutation_index];
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

}  // namespace permute

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_COMMON_PRIMITIVE_PERMUTE_H_
