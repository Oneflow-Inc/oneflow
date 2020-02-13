#ifndef ONEFLOW_CORE_KERNEL_L1_L2_REGULARIZE_GRADIENT_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_L1_L2_REGULARIZE_GRADIENT_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct L1L2RegularizeGradientKernelUtil {
  static void RegularizeGradient(DeviceCtx* ctx, int64_t n, const T* model, const T* model_diff,
                                 T* out, T l1, T l2);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_L1_L2_REGULARIZE_GRADIENT_KERNEL_UTIL_H_
