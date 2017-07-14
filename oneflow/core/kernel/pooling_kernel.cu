#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<typename FloatingPointType>
void PoolingKernel<DeviceType::kGPU, FloatingPointType>::Forward(
    const KernelCtx&,
    std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  TODO();
}

template<typename FloatingPointType>
void PoolingKernel<DeviceType::kGPU, FloatingPointType>::Backward(
    const KernelCtx&,
    std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  TODO();
}

INSTANTIATE_GPU_KERNEL_CLASS(PoolingKernel);

REGISTER_GPU_KERNEL(OperatorConf::kPoolingConf, PoolingKernel);

}  // namespace oneflow
