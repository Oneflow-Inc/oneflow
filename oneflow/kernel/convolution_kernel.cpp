#include "kernel/convolution_kernel.h"
#include <string>

namespace oneflow {

template<FloatingPointType floating_point_type>
void ConvolutionKernel<DeviceType::kCPU, floating_point_type>::Forward(
    std::function<Blob*(const std::string& )> bn_in_op2blob_ptr) {
  TODO();
}

template<FloatingPointType floating_point_type>
void ConvolutionKernel<DeviceType::kCPU, floating_point_type>::Backward(
  std::function<Blob*(const std::string& )> bn_in_op2blob_ptr) {
  TODO();
}

INSTANTIATE_CPU_KERNEL_CLASS(ConvolutionKernel);
REGISTER_KERNEL(OperatorConf::kConvolutionConf, ConvolutionKernel);

}  // namespace oneflow
