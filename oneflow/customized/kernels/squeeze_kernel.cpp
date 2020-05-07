#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/kernels/copy_data_content_kernel.h"

namespace oneflow {

#define REGISTER_SQUEEZE_KERNEL(D)                                                              \
  REGISTER_USER_KERNEL("squeeze")                                                               \
      .SetCreateFn<CopyDataContentKernel<DeviceType::D>>()                                      \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {                     \
        return ctx.device_type() == DeviceType::D;                                              \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, false));                      \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_SQUEEZE_KERNEL(kCPU)
REGISTER_SQUEEZE_KERNEL(kGPU)

}  // namespace oneflow
