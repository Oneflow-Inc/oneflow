#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<>
void Memcpy<DeviceType::kCPU>(DeviceCtx* ctx, void* dst, const void* src, size_t sz
#ifdef WITH_CUDA
                              ,
                              cudaMemcpyKind kind
#endif

) {
  if (dst == src) { return; }
  memcpy(dst, src, sz);
}

}  // namespace oneflow
