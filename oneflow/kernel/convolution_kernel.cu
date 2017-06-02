#include "kernel/convolution_kernel.h"
#include <string>

namespace oneflow {

template<FloatingPointType floating_point_type>
void ConvolutionKernel<DeviceType::kGPU, floating_point_type>::Forward(
    std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  TODO();
}

template<FloatingPointType floating_point_type>
void ConvolutionKernel<DeviceType::kGPU, floating_point_type>::Backward(
  std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  TODO();
}

INSTANTIATE_GPU_KERNEL_CLASS(ConvolutionKernel);

}  // namespace oneflow
