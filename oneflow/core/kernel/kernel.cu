#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void GpuSetSingleValue(const T value, T* addr) {
  *addr = value;
}

}  // namespace

#define INSTANTIATE_CPU_KERNEL_IF_MEMBER_FUNCTION(T, proto_t)                              \
  template<>                                                                               \
  template<>                                                                               \
  void KernelIf<DeviceType::kGPU>::PutTotalInstanceNumIntoBlob(                            \
      DeviceCtx* ctx, const int32_t instance_num, T* instance_num_ptr) const {             \
    GpuSetSingleValue<T>                                                                   \
        <<<1, 1, 0, ctx->cuda_stream()>>>(static_cast<T>(instance_num), instance_num_ptr); \
  }

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CPU_KERNEL_IF_MEMBER_FUNCTION, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
