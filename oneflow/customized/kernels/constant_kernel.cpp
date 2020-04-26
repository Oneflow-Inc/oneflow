#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/customized/kernels/op_kernel_state_wrapper.h"

namespace oneflow {
namespace user_op {

template<DeviceType device_type, typename T>
class ConstantKernel final : public OpKernel {
 public:
  ConstantKernel() = default;
  ~ConstantKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<OpKernelStateWrapper<bool>>(false);
  }
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* is_init_ = dynamic_cast<OpKernelStateWrapper<bool>*>(state);
    if (is_init_->Get()) { return; }
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    bool is_floating_value = ctx->GetAttr<bool>("is_floating_value");
    const int64_t elem_cnt = out_tensor->shape().elem_cnt();
    CHECK(elem_cnt);
    NewKernelUtil<device_type>::Fill(ctx->device_ctx(), elem_cnt,
                                     is_floating_value
                                         ? static_cast<T>(ctx->GetAttr<double>("floating_value"))
                                         : static_cast<T>(ctx->GetAttr<int64_t>("integer_value")),
                                     out_tensor->mut_dptr<T>());
    *is_init_->Mutable() = true;
  }
};

#define REGISTER_CONSTANT_XPU_KERNEL(device, dtype)                                   \
  REGISTER_USER_KERNEL("constant")                                                    \
      .SetCreateFn<ConstantKernel<device, dtype>>()                                   \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const auto& data_type = ctx.GetAttr<DataType>("dtype");                       \
        return ctx.device_type() == device && data_type == GetDataType<dtype>::value; \
      });

#define REGISTER_CONSTANT_KERNEL(dtype)                 \
  REGISTER_CONSTANT_XPU_KERNEL(DeviceType::kCPU, dtype) \
  REGISTER_CONSTANT_XPU_KERNEL(DeviceType::kGPU, dtype)

REGISTER_CONSTANT_KERNEL(float)
REGISTER_CONSTANT_KERNEL(double)
REGISTER_CONSTANT_KERNEL(int32_t)
REGISTER_CONSTANT_KERNEL(int64_t)

}  // namespace user_op
}  // namespace oneflow
