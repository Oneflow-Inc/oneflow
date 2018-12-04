#include "oneflow/core/kernel/center_loss_kernel.h"

namespace oneflow {

template <DeviceType device_type, typename PredType, typename LabelType>
void CenterLossKernel<device_type, PredType, LabelType>::VirtualLossForwardDataContent(const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // Forward

  // Backward

}

template<DeviceType device_type, typename PredType, typename LabelType>
const LossKernelConf& CenterLossKernel<device_type, PredType, LabelType>::GetLossKernelConf(
    const KernelConf& kernel_conf) const {
  return kernel_conf.center_loss_conf().loss_conf();
}

}