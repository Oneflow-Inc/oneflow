#include "oneflow/core/kernel/return_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ReturnKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kReturnConf, DeviceType::kCPU,
                            ReturnKernel<DeviceType::kCPU>);
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kReturnConf, DeviceType::kGPU,
                            ReturnKernel<DeviceType::kGPU>);

}  // namespace oneflow
