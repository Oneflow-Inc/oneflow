#include "oneflow/core/kernel/copy_comm_net_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
void CopyCommNetKernel<device_type, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  TODO();
}

template<DeviceType device_type, typename FloatingPointType>
void CopyCommNetKernel<device_type, FloatingPointType>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  TODO();
}

INSTANTIATE_KERNEL_CLASS(CopyCommNetKernel);
REGISTER_CPU_KERNEL(OperatorConf::kCopyCommNetConf, CopyCommNetKernel);

}  // namespace oneflow
