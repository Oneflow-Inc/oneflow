#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<DeviceType device_type>
class ZeroLikeKernel final : public user_op::OpKernel {
 public:
  ZeroLikeKernel() = default;
  ~ZeroLikeKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    Memset<device_type>(ctx->device_ctx(), out->mut_dptr(), 0,
                        out->shape().elem_cnt() * GetSizeOfDataType(out->data_type()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ZERO_LIKE_KERNEL(device_type_v)    \
  REGISTER_USER_KERNEL("zero_like")                 \
      .SetCreateFn<ZeroLikeKernel<device_type_v>>() \
      .SetIsMatchedHob(user_op::HobDeviceType() == device_type_v);

REGISTER_ZERO_LIKE_KERNEL(DeviceType::kCPU)
REGISTER_ZERO_LIKE_KERNEL(DeviceType::kGPU)

}  // namespace oneflow
