#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

template<>
void Memcpy<DeviceType::kGPU>(DeviceCtx* ctx, void* dst, const void* src, size_t sz,
                              cudaMemcpyKind kind) {
  if (dst == src) { return; }
  CudaCheck(cudaMemcpyAsync(dst, src, sz, kind, ctx->cuda_stream()));
}

template<>
void Memset<DeviceType::kGPU>(DeviceCtx* ctx, void* dst, const char value, size_t sz) {
  CudaCheck(cudaMemsetAsync(dst, value, sz, ctx->cuda_stream()));
}

}  // namespace oneflow
