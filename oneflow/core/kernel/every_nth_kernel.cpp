#include "oneflow/core/kernel/every_nth_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void EveryNthKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
}

template<DeviceType device_type>
void EveryNthKernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kEveryNthConf, DeviceType::kCPU,
                            EveryNthKernel<DeviceType::kCPU>);
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kEveryNthConf, DeviceType::kGPU,
                            EveryNthKernel<DeviceType::kGPU>);

}  // namespace oneflow
