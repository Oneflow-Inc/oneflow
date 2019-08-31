#include "oneflow/core/kernel/switch_output_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void SwitchOutputKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  std::string ibn = "in_";
  ibn += *BnInOp2Blob("in_index")->dptr<int32_t>();
  BnInOp2Blob("out")->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob(ibn));
}

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kSwitchOutputConf, DeviceType::kCPU,
                            SwitchOutputKernel<DeviceType::kCPU>);
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kSwitchOutputConf, DeviceType::kGPU,
                            SwitchOutputKernel<DeviceType::kGPU>);

}  // namespace oneflow
