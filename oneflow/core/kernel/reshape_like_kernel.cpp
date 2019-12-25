#include "oneflow/core/kernel/reshape_like_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ReshapeLikeKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("x");
  Blob* out_blob = BnInOp2Blob("y");
  out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kReshapeLikeConf, ReshapeLikeKernel);

}  // namespace oneflow
