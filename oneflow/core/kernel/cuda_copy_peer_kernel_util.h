#ifndef ONEFLOW_CORE_KERNEL_CUDA_COPY_PEER_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_CUDA_COPY_PEER_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

struct CudaCopyPeerKernelUtil {
  static void CopyAsync(void* dst, void* buf, const void* src, int32_t* step_mutex, size_t size,
                        cudaStream_t read, cudaStream_t write);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CUDA_COPY_PEER_KERNEL_UTIL_H_
