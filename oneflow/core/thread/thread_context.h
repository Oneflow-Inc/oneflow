#ifndef ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_
#define ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

struct ThreadCtx {
  ThreadCtx() : copy_hd_cuda_stream(nullptr) {}

  const cudaStream_t* copy_hd_cuda_stream;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_
