#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/kernels/copy_data_content_kernel.h"

namespace oneflow {

#define REGISTER_EXPAND_DIMS_KERNEL(T, D)                                                       \
  REGISTER_USER_KERNEL("expand_dims")                                                           \
      .SetCreateFn([](oneflow::user_op::KernelInitContext* ctx) {                               \
        return new CopyDataContentKernel<T, DeviceType::D>(ctx);                                \
      })                                                                                        \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {                     \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);         \
        return ctx.device_type() == DeviceType::D                                               \
               && out_desc->data_type() == GetDataType<T>::value;                               \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_EXPAND_DIMS_KERNEL(float, kCPU)
REGISTER_EXPAND_DIMS_KERNEL(double, kCPU)
REGISTER_EXPAND_DIMS_KERNEL(int8_t, kCPU)
REGISTER_EXPAND_DIMS_KERNEL(int32_t, kCPU)
REGISTER_EXPAND_DIMS_KERNEL(int64_t, kCPU)
REGISTER_EXPAND_DIMS_KERNEL(float, kGPU)
REGISTER_EXPAND_DIMS_KERNEL(double, kGPU)
REGISTER_EXPAND_DIMS_KERNEL(int8_t, kGPU)
REGISTER_EXPAND_DIMS_KERNEL(int32_t, kGPU)
REGISTER_EXPAND_DIMS_KERNEL(int64_t, kGPU)

}  // namespace oneflow
