#ifndef ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_
#define ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

struct ThreadCtx {
  ThreadCtx()
#ifdef WITH_CUDA
      : copy_hd_cuda_stream(nullptr)
#endif
  {
  }

#ifdef WITH_CUDA
  const cudaStream_t* copy_hd_cuda_stream;
#endif
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_
