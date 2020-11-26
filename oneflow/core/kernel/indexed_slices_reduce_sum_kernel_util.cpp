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
#include "oneflow/core/kernel/indexed_slices_reduce_sum_kernel_util.h"
#include "oneflow/core/kernel/unique_kernel_util.h"
#include "oneflow/core/kernel/unsorted_segment_sum_kernel_util.h"

namespace oneflow {

template<typename IDX>
int64_t GetUniqueIdxSize(int64_t n) {
  return GetCudaAlignedSize(n * sizeof(IDX));
}

template<DeviceType device_type, typename K, typename T, typename IDX>
void IndexedSlicesReduceSumKernelUtil<device_type, K, T, IDX>::ReduceSum(
    DeviceCtx* ctx, int64_t n, int64_t m, const K* indices, const T* values,
    IDX* num_unique_indices, K* indices_out, T* values_out, void* workspace,
    int64_t workspace_size_in_bytes) {
  const int64_t unique_idx_size = GetUniqueIdxSize<IDX>(n);
  CHECK_LE(unique_idx_size, workspace_size_in_bytes);
  IDX* unique_idx_ptr = reinterpret_cast<IDX*>(workspace);
  void* unique_workspace_ptr = reinterpret_cast<unsigned char*>(workspace) + unique_idx_size;
  const int64_t unique_workspace_size = workspace_size_in_bytes - unique_idx_size;
  UniqueKernelUtil<device_type, K, IDX>::Unique(ctx, n, indices, num_unique_indices, indices_out,
                                                unique_idx_ptr, unique_workspace_ptr,
                                                unique_workspace_size);
  const Shape flat_in_shape({1, n, m});
  Memset<device_type>(ctx, values_out, 0, n * m * sizeof(T));

  UnsortedSegmentSumKernelUtil<device_type, T, IDX, T>::UnsortedSegmentSum(
      ctx, unique_idx_ptr, values, n, n, 1, m, 0, values_out);
}

template<DeviceType device_type, typename K, typename T, typename IDX>
void IndexedSlicesReduceSumKernelUtil<device_type, K, T, IDX>::GetReduceSumWorkspaceSizeInBytes(
    DeviceCtx* ctx, int64_t n, int64_t m, int64_t* workspace_size_in_bytes) {
  int64_t unique_workspace_size;
  UniqueKernelUtil<device_type, K, int64_t>::GetUniqueWorkspaceSizeInBytes(ctx, n,
                                                                           &unique_workspace_size);
  *workspace_size_in_bytes = GetUniqueIdxSize<IDX>(n) + unique_workspace_size;
}

#define INSTANTIATE_INDEXED_SLICES_REDUCE_SUM_KERNEL_UTIL(device_type, key_type_pair,            \
                                                          val_type_pair, idx_type_pair)          \
  template struct IndexedSlicesReduceSumKernelUtil<device_type, OF_PP_PAIR_FIRST(key_type_pair), \
                                                   OF_PP_PAIR_FIRST(val_type_pair),              \
                                                   OF_PP_PAIR_FIRST(idx_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_INDEXED_SLICES_REDUCE_SUM_KERNEL_UTIL, DEVICE_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INSTANTIATE_INDEXED_SLICES_REDUCE_SUM_KERNEL_UTIL

}  // namespace oneflow
