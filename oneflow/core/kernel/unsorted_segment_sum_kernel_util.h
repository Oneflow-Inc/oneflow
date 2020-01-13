#ifndef ONEFLOW_CORE_KERNEL_UNSORTED_SEGMENT_SUM_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_UNSORTED_SEGMENT_SUM_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
struct UnsortedSegmentSumKernelUtil final {
  static void UnsortedSegmentSum(DeviceCtx* ctx, const K* segment_ids, const T* data,
                                 int64_t num_segment_ids, int64_t num_segments,
                                 int64_t outer_dim_size, int64_t inner_dim_size,
                                 int64_t segment_id_offset, T* out);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_UNSORTED_SEGMENT_SUM_KERNEL_UTIL_H_
