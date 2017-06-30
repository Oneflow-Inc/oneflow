#ifndef ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_
#define ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_

#include "oneflow/core/common/channel.h"

namespace oneflow {

struct ThreadCtx {
  ThreadCtx() : cpu_stream(nullptr), copy_hd_cuda_stream(nullptr) {}

  Channel<std::function<void()>>* cpu_stream;
  const cudaStream_t* copy_hd_cuda_stream;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_
