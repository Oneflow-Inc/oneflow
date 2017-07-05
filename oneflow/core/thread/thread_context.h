#ifndef ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_
#define ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_

#include "oneflow/core/device/cpu_stream.h"

namespace oneflow {

struct ThreadCtx {
  ThreadCtx() : cpu_stream(nullptr), copy_hd_cuda_stream(nullptr) {}

  CpuStream* cpu_stream;
  const cudaStream_t* copy_hd_cuda_stream;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_
