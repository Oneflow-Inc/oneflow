#ifndef ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_
#define ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_

#include "oneflow/core/device/cuda_stream_handle.h"

namespace oneflow {

struct ThreadCtx {
  void* buf_ptr;
  size_t buf_size;
#ifdef WITH_CUDA
  std::unique_ptr<CudaStreamHandle> compute_cuda_stream;
  std::unique_ptr<CudaStreamHandle> copy_h2d_cuda_stream;
  std::unique_ptr<CudaStreamHandle> copy_d2h_cuda_stream;
#endif
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_
