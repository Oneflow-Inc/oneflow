#ifndef ONEFLOW_CORE_KERNEL_BIAS_ADD_USER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BIAS_ADD_USER_KERNEL_H_

#include "oneflow/core/framework/framework.h"

namespace oneflow {
template<DeviceType device_type, typename T>
struct BiasAddUtil {
  static void BiasAdd(DeviceCtx* ctx, int64_t outer_size, int64_t bias_size, int64_t inner_size,
                      const T* x, const T* bias, T* y);
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_KERNEL_BIAS_ADD_USER_KERNEL_H_
