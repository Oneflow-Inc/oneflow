#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

template <typename T>
void LogicalNot(DeviceCtx *ctx, const int64_t n, const T *x, T *y) {
  for (int64_t i = 0; i != n; ++i) {
    y[i] = !static_cast<bool>(x[i]);
  }
}

template <DeviceType device_type, typename T>
class LogicalNotKernel final : public user_op::OpKernel {
public:
  LogicalNotKernel() = default;
  ~LogicalNotKernel() = default;

private:
  void Compute(user_op::KernelComputeContext *ctx) const override {
    const user_op::Tensor *in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor *out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    LogicalNot<T>(ctx->device_ctx(),
           in_tensor->shape().elem_cnt(),
           in_tensor->dptr<T>(),
           out_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_RELU_KERNEL(device, dtype)               \
  REGISTER_USER_KERNEL("logical_not")                     \
      .SetCreateFn<LogicalNotKernel<device, dtype>>()     \
      .SetIsMatchedHob(                                   \
          (user_op::HobDeviceTag() == device) &           \
          (user_op::HobDataType("out", 0)                 \
            == GetDataType<dtype>::value));

REGISTER_RELU_KERNEL(DeviceType::kCPU, int8_t)
REGISTER_RELU_KERNEL(DeviceType::kGPU, int8_t)
} // namespace

} // namespace oneflow