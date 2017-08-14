#include "oneflow/core/kernel/log_kernel.h"

namespace oneflow {

template<typename FloatingPointType>
void LogKernel<DeviceType::kCPU, FloatingPointType>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {}

INSTANTIATE_CPU_KERNEL_CLASS(LogKernel);
REGISTER_CPU_KERNEL(OperatorConf::kLogConf, LogKernel);

}  // namespace oneflow
