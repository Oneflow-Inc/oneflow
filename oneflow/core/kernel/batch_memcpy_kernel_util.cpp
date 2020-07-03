#include "oneflow/core/kernel/batch_memcpy_kernel_util.h"

namespace oneflow {

template<>
void BatchMemcpyKernelUtil<DeviceType::kCPU>::Copy(DeviceCtx* ctx,
                                                   const std::vector<MemcpyParam>& params) {
  for (const MemcpyParam& param : params) {
    Memcpy<DeviceType::kCPU>(ctx, param.dst, param.src, param.count);
  }
}

}  // namespace oneflow
