#include "oneflow/core/kernel/convolution_kernel.h"

namespace oneflow {

template<typename FloatingPointType>
void ConvolutionKernel<DeviceType::kCPU, FloatingPointType>::Forward(
    const KernelCtx&,
    std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  TODO();
}

template<typename FloatingPointType>
void ConvolutionKernel<DeviceType::kCPU, FloatingPointType>::Backward(
    const KernelCtx&,
    std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  TODO();
}

INSTANTIATE_CPU_KERNEL_CLASS(ConvolutionKernel);
REGISTER_CPU_KERNEL(OperatorConf::kConvolutionConf, ConvolutionKernel);

}  // namespace oneflow
