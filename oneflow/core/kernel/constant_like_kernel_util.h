#ifndef ONEFLOW_CORE_KERNEL_CONSTANT_LIKE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CONSTANT_LIKE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct ConstantLikeUtil {
  static void Forward(DeviceCtx* ctx, const int64_t elem_cnt, const T scalar, T* out_ptr);
};

}  // namespace oneflow

#endif
