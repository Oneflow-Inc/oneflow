#include "oneflow/core/kernel/convolution_kernel.h"
#include <string>

namespace oneflow {

template<typename floating_point_type>
void ConvolutionKernel<DeviceType::kGPU, floating_point_type>::Forward(
    const KernelCtx&,
    std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  TODO();
}

template<typename floating_point_type>
void ConvolutionKernel<DeviceType::kGPU, floating_point_type>::Backward(
    const KernelCtx&,
    std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  TODO();
}

INSTANTIATE_GPU_KERNEL_CLASS(ConvolutionKernel);

REGISTER_GPU_KERNEL(OperatorConf::kConvolutionConf, ConvolutionKernel);

}  // namespace oneflow
