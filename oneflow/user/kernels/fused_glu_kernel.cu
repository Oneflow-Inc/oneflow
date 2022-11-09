#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ep/common/primitive/unary_functor.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"

#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000
#include "oneflow/core/device/cuda_pseudo_bfloat16.h"

namespace oneflow {

namespace {
    
}

template<typename T>

class GpuFusedGluKernel final : public user_op::OpKernel {
 public:
  GpuFusedGluKernel() = default;
  ~GpuFusedGluKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {

  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_FUSED_GEGLU_KERNEL(dtype)                          \
  REGISTER_USER_KERNEL("fused_glu")                                     \
      .SetCreateFn<GpuFusedGluKernel<dtype>>()                          \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)  \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_GPU_FUSED_GEGLU_KERNEL(float)
REGISTER_GPU_FUSED_GEGLU_KERNEL(double)
REGISTER_GPU_FUSED_GEGLU_KERNEL(half)
REGISTER_GPU_FUSED_GEGLU_KERNEL(nv_bfloat16)

} // namespace oneflow