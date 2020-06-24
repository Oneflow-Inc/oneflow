#ifndef ONEFLOW_CORE_KERNEL_BATCH_MEMCPY_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_BATCH_MEMCPY_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

struct MemcpyParam {
  void* dst;
  const void* src;
  size_t count;
};

template<DeviceType device_type>
struct BatchMemcpyKernelUtil final {
  static void Copy(DeviceCtx* ctx, const std::vector<MemcpyParam>& params);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BATCH_MEMCPY_KERNEL_UTIL_H_
