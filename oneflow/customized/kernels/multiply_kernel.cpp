#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class MultiplyKernel final : public user_op::OpKernel {
 public:
  MultiplyKernel() = default;
  ~MultiplyKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t elem_cnt = x->shape().elem_cnt();
    CHECK_EQ(y->shape().elem_cnt(), elem_cnt);
    CHECK_EQ(out->shape().elem_cnt(), elem_cnt);
    KernelUtil<device_type, T>::Mul(ctx->device_ctx(), elem_cnt, x->dptr<T>(), y->dptr<T>(),
                                    out->mut_dptr<T>());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_MULTIPLY_KERNEL(device, dtype_pair)                                            \
  REGISTER_USER_KERNEL("multiply")                                                              \
      .SetCreateFn<MultiplyKernel<device, OF_PP_PAIR_FIRST(dtype_pair)>>()                      \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                              \
        const user_op::TensorDesc* x_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0);             \
        return ctx.device_type() == device                                                      \
               && x_desc->data_type() == OF_PP_PAIR_SECOND(dtype_pair);                         \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "x", 0, true));                        \
        return Maybe<void>::Ok();                                                               \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MULTIPLY_KERNEL, DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)
#undef REGISTER_MULTIPLY_KERNEL

}  // namespace oneflow
