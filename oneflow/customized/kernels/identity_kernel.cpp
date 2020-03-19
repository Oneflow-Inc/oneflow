#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<DeviceType device_type>
class IdentityOpKernel final : public user_op::OpKernel {
 public:
  IdentityOpKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  IdentityOpKernel() = default;
  ~IdentityOpKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    Memcpy<device_type>(ctx->device_ctx(), out->mut_dptr(), in->dptr(),
                        out->shape().elem_cnt() * GetSizeOfDataType(out->data_type()));
  };
};

#define REGISTER_IDENTITY_OP_KERNEL(device_type_v)                                              \
  REGISTER_USER_KERNEL("identity")                                                              \
      .SetCreateFn([](user_op::KernelInitContext* ctx) {                                        \
        return new IdentityOpKernel<device_type_v>(ctx);                                        \
      })                                                                                        \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) -> bool {             \
        return ctx.device_type() == device_type_v;                                              \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, false));                      \
        return Maybe<void>::Ok();                                                               \
      });                                                                                       \
  ;

REGISTER_IDENTITY_OP_KERNEL(DeviceType::kCPU)
REGISTER_IDENTITY_OP_KERNEL(DeviceType::kGPU)

}  // namespace oneflow
