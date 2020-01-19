#ifndef ONEFLOW_CORE_KERNEL_REGULARIZE_GRADIENT_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_REGULARIZE_GRADIENT_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct RegularizeGradientKernelUtil {
  static void RegularizeGradient(DeviceCtx* ctx, int64_t n, const T* model, const T* model_diff,
                                 T* out, const T l1_scale, const T l2_scale);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_REGULARIZE_GRADIENT_KERNEL_UTIL_H_
