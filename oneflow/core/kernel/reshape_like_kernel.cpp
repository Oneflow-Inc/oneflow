#include "oneflow/core/kernel/reshape_like_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ReshapeLikeKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("x");
  Blob* out_blob = BnInOp2Blob("y");
  out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
}

template<DeviceType device_type>
void ReshapeLikeKernel<device_type>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("y")->CopyDim0ValidNumFrom(ctx.device_ctx, BnInOp2Blob("like"));
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kReshapeLikeConf, ReshapeLikeKernel);

}  // namespace oneflow
