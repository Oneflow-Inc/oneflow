#ifndef ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_
#define ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_

#include "oneflow/core/device/cuda_stream_handle.h"

namespace oneflow {

struct ThreadCtx {
#ifdef WITH_CUDA
  std::unique_ptr<CudaStreamHandle> g_cuda_stream;
  Channel<CudaCBEvent>* cb_event_chan;
#endif
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_
