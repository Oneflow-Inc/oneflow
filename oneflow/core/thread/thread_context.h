#ifndef ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_
#define ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

struct ThreadCtx {
  ThreadCtx()
#ifdef WITH_CUDA
      : copy_h2d_cuda_stream(nullptr),
        copy_d2h_cuda_stream(nullptr)
#endif
  {
  }

#ifdef WITH_CUDA
  const cudaStream_t* copy_h2d_cuda_stream;
  const cudaStream_t* copy_d2h_cuda_stream;
#endif
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_
