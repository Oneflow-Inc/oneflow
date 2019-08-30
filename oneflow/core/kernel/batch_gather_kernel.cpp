#include "oneflow/core/kernel/batch_gather_kernel.h"
#include "oneflow/core/kernel/batch_gather_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
const PbMessage& BatchGatherKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().batch_gather_conf();
}

template<DeviceType device_type, typename T>
void BatchGatherKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BatchGatherKernelUtil<device_type, T>::Forward(ctx.device_ctx, BnInOp2Blob("in"),
                                                 BnInOp2Blob("indices"), BnInOp2Blob("out"));
}

REGISTER_KERNEL_HELPER_GPU_FLOATING(OperatorConf::kBatchGatherConf, BatchGatherKernel);

}  // namespace oneflow
