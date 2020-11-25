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
#include "oneflow/core/kernel/unsorted_segment_sum_kernel_util.h"

namespace oneflow {

template<typename T, typename K>
struct UnsortedSegmentSumKernelUtil<DeviceType::kCPU, T, K, T> final {
  static void UnsortedSegmentSum(DeviceCtx* ctx, const K* segment_ids, const T* data,
                                 int64_t num_segment_ids, int64_t num_segments,
                                 int64_t outer_dim_size, int64_t inner_dim_size,
                                 int64_t segment_id_offset, T* out);
};

template<typename T, typename K>
void UnsortedSegmentSumKernelUtil<DeviceType::kCPU, T, K, T>::UnsortedSegmentSum(
    DeviceCtx* ctx, const K* segment_ids, const T* data, int64_t num_segment_ids,
    int64_t num_segments, int64_t outer_dim_size, int64_t inner_dim_size, int64_t segment_id_offset,
    T* out) {
  FOR_RANGE(int64_t, outer_idx, 0, outer_dim_size) {
    FOR_RANGE(int64_t, i, 0, num_segment_ids) {
      CHECK_GE(segment_ids[i], 0);
      const int64_t idx = segment_ids[i] - segment_id_offset;
      T* to = out + outer_idx * num_segments * inner_dim_size + idx * inner_dim_size;
      if (idx >= 0 && idx < num_segments) {
        const T* from = data + outer_idx * num_segment_ids * inner_dim_size + i * inner_dim_size;
        std::transform(from, from + inner_dim_size, to, to, std::plus<T>());
      }
    }
  }
}
#define INITIATE_UNSORTED_SEGMENT_SUM_KERNEL_UTIL_CPU(in_type_pair, index_type_pair)             \
  template struct UnsortedSegmentSumKernelUtil<DeviceType::kCPU, OF_PP_PAIR_FIRST(in_type_pair), \
                                               OF_PP_PAIR_FIRST(index_type_pair),                \
                                               OF_PP_PAIR_FIRST(in_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_UNSORTED_SEGMENT_SUM_KERNEL_UTIL_CPU,
                                 UNSORTED_SEGMENT_SUM_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);

#undef INITIATE_UNSORTED_SEGMENT_SUM_KERNEL_UTIL_CPU

}  // namespace oneflow
