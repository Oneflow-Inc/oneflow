#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ScalarAddUserKernel final : public user_op::OpKernel {
 public:
  ScalarAddUserKernel() = default;
  ~ScalarAddUserKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T scalar_operand = 0;
    if (ctx->Attr<bool>("has_int_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<int64_t>("int_operand"));
    } else if (ctx->Attr<bool>("has_float_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<double>("float_operand"));
    } else {
      UNIMPLEMENTED();
    }
    const T* in_ptr = in->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();

    KernelUtil<device_type, T>::AddByScalar(ctx->device_ctx(), out->shape().elem_cnt(), in_ptr,
                                            scalar_operand, out_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_KERNEL(kernel_device_type, dtype)                                    \
  REGISTER_USER_KERNEL("scalar_add")                                                  \
      .SetCreateFn<ScalarAddUserKernel<DeviceType::k##kernel_device_type, dtype>>()   \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0); \
        return ctx.device_type() == DeviceType::k##kernel_device_type                 \
               && y_desc->data_type() == GetDataType<dtype>::value;                   \
      });

REGISTER_KERNEL(CPU, int8_t)
REGISTER_KERNEL(CPU, int32_t)
REGISTER_KERNEL(CPU, int64_t)
REGISTER_KERNEL(CPU, float)
REGISTER_KERNEL(CPU, double)
REGISTER_KERNEL(GPU, int8_t)
REGISTER_KERNEL(GPU, int32_t)
REGISTER_KERNEL(GPU, int64_t)
REGISTER_KERNEL(GPU, float)
REGISTER_KERNEL(GPU, double)

}  // namespace oneflow
