#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/kernel_registration.h"
#include "oneflow/core/kernel/kernel_registration.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace user_op {

class ConstState final : public OpKernelState {
 public:
  ConstState(bool is_init) : is_init_(is_init) {}
  ~ConstState() = default;
  bool is_inited() const { return is_init_; }
  void set_is_inited(bool val) { is_init_ = val; }

 private:
  bool is_init_;
};

template<DeviceType device_type, typename T>

class ConstantLikeKernel final : public user_op::OpKernel {
 public:
  ConstantLikeKernel() = default;
  ~ConstantLikeKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<ConstState>(false);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* const_state = dynamic_cast<ConstState*>(state);
    CHECK(ctx->all_outputs_constant());
    if (const_state->is_inited()) { return; }
    
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    bool is_floating_value = ctx->GetAttr<bool>("is_floating_value");

    auto value = is_floating_value ? ctx->GetAttr<double>("floating_value")
                                   : ctx->GetAttr<int64_t>("integer_value");

    NewKernelUtil<device_type>::Fill(ctx->device_ctx(), out->shape().elem_cnt(),
                                     static_cast<T>(value), out->mut_dptr<T>());
    const_state->set_is_inited(true);                              
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CONSTANT_LIKE_KERNEL_WITH_DEVICE_AND_DTYPE(device, dtype)            \
  REGISTER_USER_KERNEL("constant_like")                                               \
      .SetCreateFn<ConstantLikeKernel<device, dtype>>()                               \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const auto& data_type = ctx.GetAttr<DataType>("dtype");                       \
        return ctx.device_type() == device && data_type == GetDataType<dtype>::value; \
      });

#define REGISTER_CONSTANT_LIKE_KERNEL(dtype)                                   \
  REGISTER_CONSTANT_LIKE_KERNEL_WITH_DEVICE_AND_DTYPE(DeviceType::kCPU, dtype) \
  REGISTER_CONSTANT_LIKE_KERNEL_WITH_DEVICE_AND_DTYPE(DeviceType::kGPU, dtype)

REGISTER_CONSTANT_LIKE_KERNEL(float);
REGISTER_CONSTANT_LIKE_KERNEL(double);
REGISTER_CONSTANT_LIKE_KERNEL(int8_t);
REGISTER_CONSTANT_LIKE_KERNEL(int32_t);
REGISTER_CONSTANT_LIKE_KERNEL(int64_t);

#undef REGISTER_CONSTANT_LIKE_KERNEL

}  // namespace user_op

}  // namespace oneflow