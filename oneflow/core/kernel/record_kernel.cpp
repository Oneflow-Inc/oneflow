#include "oneflow/core/kernel/record_kernel.h"

namespace oneflow {

template<typename FloatingPointType>
void RecordKernel<DeviceType::kCPU, FloatingPointType>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {}

INSTANTIATE_CPU_KERNEL_CLASS(RecordKernel);
REGISTER_CPU_KERNEL(OperatorConf::kRecordConf, RecordKernel);

}  // namespace oneflow
