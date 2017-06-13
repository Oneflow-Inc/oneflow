#ifndef ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_
#define ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_

#include "oneflow/core/common/channel.h"

namespace oneflow {

struct ThreadContext {
  ThreadContext() : cpu_stream_(nullptr),
                    copy_hd_cuda_stream(nullptr),
                    compute_cuda_stream(nullptr) {}
  
  Channel<std::function<void()>>* cpu_stream_;
  const cudaStream_t* copy_hd_cuda_stream;
  const cudaStream_t* compute_cuda_stream;
};

} // namespace oneflow

#endif // ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_
