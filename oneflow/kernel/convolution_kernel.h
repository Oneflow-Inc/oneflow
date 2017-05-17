#ifndef ONEFLOW_KERNEL_CONVOLUTION_KERNEL_H_
#define ONEFLOW_KERNEL_CONVOLUTION_KERNEL_H_

#include "kernel/kernel.h"
#include "kernel/kernel_manager.h"
#include "job/resource.pb.h"
#include "job/job_conf.pb.h"

namespace oneflow {

template<DeviceType device_type, FloatingPointType floating_point_type>
class ConvolutionKernel final {
};

template<FloatingPointType floating_point_type>
class ConvolutionKernel<DeviceType::kCPU, floating_point_type> final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvolutionKernel);
  ConvolutionKernel() = default;
  ~ConvolutionKernel() = default;

  void Forward(std::function<Blob*(const std::string&)>) override;
  void Backward(std::function<Blob*(const std::string&)>) override;
};

template<FloatingPointType floating_point_type>
class ConvolutionKernel<DeviceType::kGPU, floating_point_type> final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvolutionKernel);
  ConvolutionKernel() = default;
  ~ConvolutionKernel() = default;

  void Forward(std::function<Blob*(const std::string&)>) override;
  void Backward(std::function<Blob*(const std::string&)>) override;
};

}  // namespace oneflow

#endif  // ONEFLOW_KERNEL_CONVOLUTION_KERNEL_H_
