#include "oneflow/core/kernel/switch_output_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void SwitchOutputKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  std::string ibn = "in_";
  ibn += *BnInOp2Blob("in_index")->dptr<int32_t>();
  BnInOp2Blob("out")->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob(ibn));
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kSwitchOutputConf, SwitchOutputKernel);

}  // namespace oneflow
