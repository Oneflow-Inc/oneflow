#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/kernels/op_kernel_state_wrapper.h"
#include "oneflow/customized/kernels/dropout_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class DropoutKernelV2 final : public user_op::OpKernel {
 public:
  DropoutKernelV2() = default;
  ~DropoutKernelV2() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    int64_t seed = ctx->Attr<int64_t>("seed");
    return std::make_shared<OpKernelStateWrapper<DropoutUtil<device_type>>>(seed,
                                                                            ctx->device_ctx());
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const float scale = ctx->Attr<float>("scale");
    const float threshold = ctx->Attr<float>("rate");
    auto* dropout_util = dynamic_cast<OpKernelStateWrapper<DropoutUtil<device_type>>*>(state);
    dropout_util->Mutable()->Dropout(in->shape().elem_cnt(), scale, threshold, in->dptr<T>(),
                                     out->mut_dptr<T>(), mask->mut_dptr<int8_t>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DROPOUT_V2_KERNEL(device_type_v, dtype_pair)                           \
  REGISTER_USER_KERNEL("dropout_v2")                                                    \
      .SetCreateFn<DropoutKernelV2<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>>()      \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                      \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0); \
        return ctx.device_type() == device_type_v                                       \
               && out_desc->data_type() == OF_PP_PAIR_SECOND(dtype_pair);               \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_DROPOUT_V2_KERNEL, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
