#ifndef ONEFLOW_CUSTOMIZED_KERNELS_TWO_STAGE_REDUCE_UTIL_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_TWO_STAGE_REDUCE_UTIL_H_

#include "oneflow/core/device/device_context.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
struct TwoStageReduceKernelUtil {
  static void DivideMaxCount(DeviceCtx* ctx, const int64_t n, const T* x, const K* max_count, T* y);
  static void ElemWiseSetWithMask(DeviceCtx* ctx, const int64_t n, const T* x, const K* mask, T* y);
  static void MulCount(DeviceCtx* ctx, const int64_t n, const T* x, const K* count, T* y);
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_TWO_STAGE_REDUCE_UTIL_H_
