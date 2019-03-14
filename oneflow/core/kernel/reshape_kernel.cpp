#include "oneflow/core/kernel/reshape_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ReshapeKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
}

template<DeviceType device_type>
void ReshapeKernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  in_diff_blob->CopyDataContentFrom(ctx.device_ctx, out_diff_blob);
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kReshapeConf, ReshapeKernel);
ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kReshapeGradConf, ReshapeKernel);

}  // namespace oneflow
