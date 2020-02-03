#ifndef ONEFLOW_CORE_KERNEL_SQUARE_SUM_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_SQUARE_SUM_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct SquareSumKernelUtil {
  static void SquareSum(DeviceCtx* ctx, int64_t n, const T* x, T* y);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SQUARE_SUM_KERNEL_UTIL_H_
