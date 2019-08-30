#include "oneflow/core/kernel/output_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void OutputKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kOutputConf, DeviceType::kCPU,
                            OutputKernel<DeviceType::kCPU>);
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kOutputConf, DeviceType::kGPU,
                            OutputKernel<DeviceType::kGPU>);

}  // namespace oneflow
