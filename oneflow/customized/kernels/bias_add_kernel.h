#ifndef ONEFLOW_CORE_KERNEL_BIAS_ADD_USER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BIAS_ADD_USER_KERNEL_H_

#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct BiasAddUtil {
  static void BiasAdd(DeviceCtx* ctx, int64_t outer_size, int64_t bias_size, int64_t inner_size,
                      const T* x, const T* bias, T* y);
};

template<DeviceType device_type, typename T>
class BiasAddUserKernel final : public user_op::OpKernel {
 public:
  BiasAddUserKernel() = default;
  ~BiasAddUserKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* a_tensor = ctx->Tensor4ArgNameAndIndex("a", 0);
    const auto* b_tensor = ctx->Tensor4ArgNameAndIndex("b", 0);
    auto* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t bias_add_axis = ctx->GetAttr<int32_t>("axis");
    const int64_t outer_size = a_tensor->shape().Count(0, bias_add_axis);
    const int64_t bias_size = a_tensor->shape().At(bias_add_axis);
    const int64_t inner_size = a_tensor->shape().Count(bias_add_axis + 1);
    BiasAddUtil<device_type, T>::BiasAdd(ctx->device_ctx(), outer_size, bias_size, inner_size,
                                         a_tensor->dptr<T>(), b_tensor->dptr<T>(),
                                         out_tensor->mut_dptr<T>());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BIAS_ADD_USER_KERNEL(op_device_type, dtype)                                    \
  REGISTER_USER_KERNEL("bias_add")                                                              \
      .SetCreateFn<BiasAddUserKernel<DeviceType::k##op_device_type, dtype>>()                   \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                              \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);         \
        return ctx.device_type() == DeviceType::k##op_device_type                               \
               && out_desc->data_type() == GetDataType<dtype>::value;                           \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "a", 0, true));                        \
        return Maybe<void>::Ok();                                                               \
      });

}  // namespace oneflow
#endif  // ONEFLOW_CORE_KERNEL_BIAS_ADD_USER_KERNEL_H_
