#ifndef ONEFLOW_CORE_KERNEL_CONVOLUTION_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CONVOLUTION_KERNEL_H_

#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename floating_point_type>
class ConvolutionKernel final {
};

template<typename floating_point_type>
class ConvolutionKernel<DeviceType::kCPU, floating_point_type> final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvolutionKernel);
  ConvolutionKernel() = default;
  ~ConvolutionKernel() = default;

  void Forward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
};

template<typename floating_point_type>
class ConvolutionKernel<DeviceType::kGPU, floating_point_type> final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvolutionKernel);
  ConvolutionKernel() = default;
  ~ConvolutionKernel() = default;

  void Forward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif // ONEFLOW_CORE_KERNEL_CONVOLUTION_KERNEL_H_
