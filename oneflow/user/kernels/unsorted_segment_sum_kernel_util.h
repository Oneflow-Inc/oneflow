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
#ifndef ONEFLOW_CORE_KERNELS_UNSORTED_SEGMENT_SUM_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNELS_UNSORTED_SEGMENT_SUM_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K, typename U>
struct UnsortedSegmentSumKernelUtil final {
  static void UnsortedSegmentSum(DeviceCtx* ctx, const K* segment_ids, const U* data,
                                 int64_t num_segment_ids, int64_t num_segments,
                                 int64_t outer_dim_size, int64_t inner_dim_size,
                                 int64_t segment_id_offset, T* out);
};

#define UNSORTED_SEGMENT_SUM_DATA_TYPE_SEQ \
  FLOATING_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNELS_UNSORTED_SEGMENT_SUM_KERNEL_UTIL_H_
