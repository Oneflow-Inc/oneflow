#include "oneflow/core/kernel/identity_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void IdentityKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

template<DeviceType device_type>
void IdentityKernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob(GenDiffBn("in"))->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob(GenDiffBn("out")));
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kParallelCastConf, IdentityKernel);

}  // namespace oneflow
