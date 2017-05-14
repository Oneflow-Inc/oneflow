#include "kernel/convolution_kernel.h"
#include <string>

namespace oneflow {

template<FloatingPointType floating_point_type>
void ConvolutionKernel<DeviceType::kGPU, floating_point_type>::Forward(
    std::function<Blob*(const std::string& bn_in_op)>) {
  TODO();
}

template<FloatingPointType floating_point_type>
void ConvolutionKernel<DeviceType::kGPU, floating_point_type>::Backward(
  std::function<Blob*(const std::string& bn_in_op)>) {
  TODO();
}

}  // namespace oneflow
