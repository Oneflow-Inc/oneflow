#ifndef ONEFLOW_CUSTOMIZED_KERNELS_TWO_STAGE_REDUCE_UTIL_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_TWO_STAGE_REDUCE_UTIL_H_

#include "oneflow/core/device/device_context.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
struct TwoStageReduceKernelUtil {
  static void Divide(DeviceCtx* ctx, const int64_t n, const T* x, const K* count, T* y);
  static void Mask(DeviceCtx* ctx, const int64_t n, const T* x, const K* mask, T* y);
  static void Scale(DeviceCtx* ctx, const int64_t n, const T* x, const K* scale, T* y);
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_TWO_STAGE_REDUCE_UTIL_H_
