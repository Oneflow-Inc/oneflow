#include "oneflow/core/kernel/every_nth_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void EveryNthKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kEveryNthConf, EveryNthKernel)

}  // namespace oneflow
