#ifndef ONEFLOW_CORE_KERNEL_KERNEL_CONTEXT_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_CONTEXT_H_

#include "oneflow/core/actor/device_context.h"

namespace oneflow {

struct KernelCtx {
  const DeviceCtx* device_ctx;
};

} // namespace oneflow

#endif // ONEFLOW_CORE_KERNEL_KERNEL_CONTEXT_H_
