#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ScalarMulUserKernel final : public user_op::OpKernel {
 public:
  ScalarMulUserKernel() = default;
  ~ScalarMulUserKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T scalar_operand = static_cast<T>(0);
    if (ctx->Attr<bool>("has_int_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<int64_t>("int_operand"));
    } else if (ctx->Attr<bool>("has_float_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<double>("float_operand"));
    } else {
      UNIMPLEMENTED();
    }
    const T* in_ptr = in->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();

    NewKernelUtil<device_type>::MulByScalar(ctx->device_ctx(), out->shape().elem_cnt(), in_ptr,
                                            scalar_operand, out_ptr);
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_KERNEL(kernel_device_type, dtype)                                   \
  REGISTER_USER_KERNEL("scalar_mul")                                                 \
      .SetCreateFn<ScalarMulUserKernel<DeviceType::k##kernel_device_type, dtype>>()  \
      .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::k##kernel_device_type \
                       & user_op::HobDataType("out", 0) == GetDataType<dtype>::value);

REGISTER_KERNEL(CPU, int32_t)
REGISTER_KERNEL(CPU, int64_t)
REGISTER_KERNEL(CPU, float)
REGISTER_KERNEL(CPU, double)
REGISTER_KERNEL(GPU, int32_t)
REGISTER_KERNEL(GPU, int64_t)
REGISTER_KERNEL(GPU, float)
REGISTER_KERNEL(GPU, double)
REGISTER_KERNEL(GPU, float16)

}  // namespace oneflow
