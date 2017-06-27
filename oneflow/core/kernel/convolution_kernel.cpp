#include "oneflow/core/kernel/convolution_kernel.h"

namespace oneflow {

template<typename floating_point_type>
void ConvolutionKernel<DeviceType::kCPU, floating_point_type>::Forward(
    const KernelCtx&,
    std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  TODO();
}

template<typename floating_point_type>
void ConvolutionKernel<DeviceType::kCPU, floating_point_type>::Backward(
    const KernelCtx&,
    std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  TODO();
}

INSTANTIATE_CPU_KERNEL_CLASS(ConvolutionKernel);
REGISTER_CPU_KERNEL(OperatorConf::kConvolutionConf, ConvolutionKernel);

}  // namespace oneflow
