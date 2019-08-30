#include "oneflow/core/kernel/batch_gather_grad_kernel.h"
#include "oneflow/core/kernel/batch_gather_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
const PbMessage& BatchGatherGradKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().batch_gather_grad_conf();
}

template<DeviceType device_type, typename T>
void BatchGatherGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BatchGatherKernelUtil<device_type, T>::Backward(ctx.device_ctx, BnInOp2Blob("out_diff"),
                                                  BnInOp2Blob("indices"), BnInOp2Blob("in_diff"));
}

template<DeviceType device_type, typename T>
void BatchGatherGradKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

REGISTER_KERNEL_HELPER_GPU_FLOATING(OperatorConf::kBatchGatherGradConf, BatchGatherGradKernel);

}  // namespace oneflow
