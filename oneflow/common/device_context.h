#ifndef ONEFLOW_COMMON_DEVICE_CONTEXT_H_
#define ONEFLOW_COMMON_DEVICE_CONTEXT_H_

#include "cuda_runtime.h"

namespace oneflow {

struct DeviceContext {
  cudaStream_t cuda_stream;
};

} // namespace oneflow

#endif // ONEFLOW_COMMON_DEVICE_CONTEXT_H_
