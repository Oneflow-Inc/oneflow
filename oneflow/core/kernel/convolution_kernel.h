#ifndef ONEFLOW_KERNEL_CONVOLUTION_KERNEL_H_
#define ONEFLOW_KERNEL_CONVOLUTION_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/conf/resource.pb.h"
#include "oneflow/core/conf/job_conf.pb.h"

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

  void Forward(std::function<Blob*(const std::string&)>) const override;
  void Backward(std::function<Blob*(const std::string&)>) const override;
};

template<FloatingPointType floating_point_type>
class ConvolutionKernel<DeviceType::kGPU, floating_point_type> final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvolutionKernel);
  ConvolutionKernel() = default;
  ~ConvolutionKernel() = default;

  void Forward(std::function<Blob*(const std::string&)>) const override;
  void Backward(std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_KERNEL_CONVOLUTION_KERNEL_H_
