#ifndef ONEFLOW_CORE_KERNEL_KERNEL_CONTEXT_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_CONTEXT_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/channel.h"

namespace oneflow {

struct KernelCtx {
  Channel<std::function<void()>>* cpu_stream;
  const cudaStream_t* cuda_stream;
};

} // namespace oneflow

#endif // ONEFLOW_CORE_KERNEL_KERNEL_CONTEXT_H_
