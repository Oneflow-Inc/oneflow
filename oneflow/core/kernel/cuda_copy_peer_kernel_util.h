#ifndef ONEFLOW_CORE_KERNEL_CUDA_COPY_PEER_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_CUDA_COPY_PEER_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

struct CudaCopyPeerCtx;

struct CudaCopyPeerKernelUtil {
  static void CopyAsync(CudaCopyPeerCtx* ctx, void* dst, const void* src, int32_t size);
  static void CtxCreate(CudaCopyPeerCtx** ctx, int32_t dst_dev_id, int32_t src_dev_id,
                        cudaStream_t recv_stream);
  static void CtxDestroy(CudaCopyPeerCtx* ctx);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CUDA_COPY_PEER_KERNEL_UTIL_H_
