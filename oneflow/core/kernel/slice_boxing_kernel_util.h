#ifndef ONEFLOW_CORE_KERNEL_SLICE_BOXING_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_SLICE_BOXING_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct SliceBoxingKernelUtil {
  static void Add(DeviceCtx* ctx, int64_t n, const T* a, const T* b, T* out);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SLICE_BOXING_KERNEL_UTIL_H_
