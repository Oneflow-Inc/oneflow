#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<typename T, DeviceType device_type>
class ExpandDimsKernel final : public user_op::OpKernel {
 public:
  ExpandDimsKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  ExpandDimsKernel() = default;
  ~ExpandDimsKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    Memcpy<device_type>(ctx->device_ctx(), out->mut_dptr<T>(), in->dptr<T>(),
                        in->shape().elem_cnt() * sizeof(T));
  };
};

#define REGISTER_EXPAND_DIMS_KERNEL(dtype)                                                      \
  REGISTER_USER_KERNEL("expand_dims")                                                           \
      .SetCreateFn([](oneflow::user_op::KernelInitContext* ctx) {                               \
        return new ExpandDimsKernel<dtype, DeviceType::kCPU>(ctx);                              \
      })                                                                                        \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {                     \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);         \
        return ctx.device_type() == DeviceType::kCPU                                            \
               && out_desc->data_type() == GetDataType<dtype>::value;                           \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });                                                                                       \
  REGISTER_USER_KERNEL("expand_dims")                                                           \
      .SetCreateFn([](oneflow::user_op::KernelInitContext* ctx) {                               \
        return new ExpandDimsKernel<dtype, DeviceType::kGPU>(ctx);                              \
      })                                                                                        \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {                     \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);         \
        return ctx.device_type() == DeviceType::kGPU                                            \
               && out_desc->data_type() == GetDataType<dtype>::value;                           \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_EXPAND_DIMS_KERNEL(float)
REGISTER_EXPAND_DIMS_KERNEL(double)
REGISTER_EXPAND_DIMS_KERNEL(int8_t)
REGISTER_EXPAND_DIMS_KERNEL(int32_t)
REGISTER_EXPAND_DIMS_KERNEL(int64_t)

}  // namespace oneflow
