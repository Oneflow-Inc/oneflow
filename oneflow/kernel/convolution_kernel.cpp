#include "kernel/convolution_kernel.h"
#include <string>
#include "kernel/kernel_manager.h"
#include "register/blob.h"

namespace oneflow {

template<FloatingPointType floating_point_type>
void ConvolutionKernel<DeviceType::kCPU, floating_point_type>::Forward(
    std::function<Blob*(const std::string& bn_in_op)>) {
  TODO();
}

template<FloatingPointType floating_point_type>
void ConvolutionKernel<DeviceType::kCPU, floating_point_type>::Backward(
  std::function<Blob*(const std::string& bn_in_op)>) {
  TODO();
}

REGISTER_CPU_KERNEL(OperatorConf::kConvolutionConf,
                          ConvolutionKernel);

}  // namespace oneflow
