#include "oneflow/core/kernel/return_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ReturnKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("in"));
  ctx.device_ctx->SyncDevice();
}

template<DeviceType device_type>
void ReturnKernel<device_type>::ForwardHeader(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyHeaderFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kReturnConf, ReturnKernel);

}  // namespace oneflow
