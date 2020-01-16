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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && CUDA_VERSION >= 10000
#define UNSORTED_SEGMENT_SUM_DATA_TYPE_SEQ \
  FLOATING_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) FLOAT16_DATA_TYPE_SEQ
#else
#define UNSORTED_SEGMENT_SUM_DATA_TYPE_SEQ \
  FLOATING_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)
#endif

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_UNSORTED_SEGMENT_SUM_KERNEL_UTIL_H_
