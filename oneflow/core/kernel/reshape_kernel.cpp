#include "oneflow/core/kernel/reshape_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ReshapeKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
}

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kReshapeConf, DeviceType::kCPU,
                            ReshapeKernel<DeviceType::kCPU>);
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kReshapeConf, DeviceType::kGPU,
                            ReshapeKernel<DeviceType::kGPU>);

}  // namespace oneflow
