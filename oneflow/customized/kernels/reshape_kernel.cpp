#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/kernels/copy_data_content_kernel.h"

namespace oneflow {

#define REGISTER_RESHAPE_KERNEL(device)                                                         \
  REGISTER_USER_KERNEL("reshape")                                                               \
      .SetCreateFn<CopyDataContentKernel<device>>()                                             \
      .SetIsMatchedPred(                                                                        \
          [](const user_op::KernelRegContext& ctx) { return ctx.device_type() == device; })     \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, false));                      \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_RESHAPE_KERNEL(DeviceType::kCPU)
REGISTER_RESHAPE_KERNEL(DeviceType::kGPU)
}  // namespace oneflow
