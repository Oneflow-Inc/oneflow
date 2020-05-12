#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class SigmoidKernel final : public user_op::OpKernel {
 public:
  SigmoidKernel() = default;
  ~SigmoidKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);
    NewKernelUtil<device_type>::Sigmoid(ctx->device_ctx(), x->shape().elem_cnt(), x->dptr<T>(),
                                        y->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SIGMOID_KERNEL(device, dtype)                                                  \
  REGISTER_USER_KERNEL("sigmoid")                                                               \
      .SetCreateFn<SigmoidKernel<device, dtype>>()                                              \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                              \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);           \
        return ctx.device_type() == device && y_desc->data_type() == GetDataType<dtype>::value; \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_SIGMOID_KERNEL(DeviceType::kCPU, float)
REGISTER_SIGMOID_KERNEL(DeviceType::kCPU, double)
REGISTER_SIGMOID_KERNEL(DeviceType::kGPU, float)
REGISTER_SIGMOID_KERNEL(DeviceType::kGPU, double)
REGISTER_SIGMOID_KERNEL(DeviceType::kGPU, float16)

template<DeviceType device_type, typename T>
class SigmoidGradKernel final : public user_op::OpKernel {
 public:
  SigmoidGradKernel() = default;
  ~SigmoidGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    NewKernelUtil<device_type>::SigmoidBackward(ctx->device_ctx(), y_blob->shape().elem_cnt(),
                                                y_blob->dptr<T>(), y_blob->dptr<T>(),
                                                dy_blob->dptr<T>(), dx_blob->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SIGMOID_GRAD_KERNEL(device, dtype)                                              \
  REGISTER_USER_KERNEL("sigmoid_grad")                                                           \
      .SetCreateFn<SigmoidGradKernel<device, dtype>>()                                           \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                               \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0);            \
        return ctx.device_type() == device && dx_desc->data_type() == GetDataType<dtype>::value; \
      })                                                                                         \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                     \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {  \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "dy", 0, true));                         \
        return Maybe<void>::Ok();                                                                \
      });

REGISTER_SIGMOID_GRAD_KERNEL(DeviceType::kCPU, float)
REGISTER_SIGMOID_GRAD_KERNEL(DeviceType::kCPU, double)
REGISTER_SIGMOID_GRAD_KERNEL(DeviceType::kGPU, float)
REGISTER_SIGMOID_GRAD_KERNEL(DeviceType::kGPU, double)
REGISTER_SIGMOID_GRAD_KERNEL(DeviceType::kGPU, float16)

}  // namespace

}  // namespace oneflow
