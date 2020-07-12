#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/kernels/copy_data_content_kernel.h"

namespace oneflow {

#define REGISTER_EXPAND_DIMS_KERNEL(D)                                                          \
  REGISTER_USER_KERNEL("expand_dims")                                                           \
      .SetCreateFn<CopyDataContentKernel<DeviceType::D>>()                                      \
      .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::D)                               \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, false));                      \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_EXPAND_DIMS_KERNEL(kCPU)
REGISTER_EXPAND_DIMS_KERNEL(kGPU)

}  // namespace oneflow
