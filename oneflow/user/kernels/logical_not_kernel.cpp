#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

template <typename T>
void LogicalNot(DeviceCtx *ctx, const int64_t n, const T *x, int8_t *y) {
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
           out_tensor->mut_dptr<int8_t>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_LOGICAL_NOT_KERNEL(device, dtype)               \
  REGISTER_USER_KERNEL("logical_not")                     \
      .SetCreateFn<LogicalNotKernel<device, dtype>>()     \
      .SetIsMatchedHob(                                   \
          (user_op::HobDeviceTag() == device) &           \
          (user_op::HobDataType("in", 0)                 \
            == GetDataType<dtype>::value));


REGISTER_LOGICAL_NOT_KERNEL(DeviceType::kCPU, int8_t);
REGISTER_LOGICAL_NOT_KERNEL(DeviceType::kCPU, int32_t);
REGISTER_LOGICAL_NOT_KERNEL(DeviceType::kCPU, int64_t);
REGISTER_LOGICAL_NOT_KERNEL(DeviceType::kCPU, float);
REGISTER_LOGICAL_NOT_KERNEL(DeviceType::kCPU, double);
REGISTER_LOGICAL_NOT_KERNEL(DeviceType::kCPU, float16);



REGISTER_LOGICAL_NOT_KERNEL(DeviceType::kGPU, int8_t);
REGISTER_LOGICAL_NOT_KERNEL(DeviceType::kGPU, int32_t);
REGISTER_LOGICAL_NOT_KERNEL(DeviceType::kGPU, int64_t);
REGISTER_LOGICAL_NOT_KERNEL(DeviceType::kGPU, float);
REGISTER_LOGICAL_NOT_KERNEL(DeviceType::kGPU, double);
REGISTER_LOGICAL_NOT_KERNEL(DeviceType::kCPU, float16);

} // namespace

} // namespace oneflow