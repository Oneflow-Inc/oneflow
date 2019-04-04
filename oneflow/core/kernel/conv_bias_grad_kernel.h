#ifndef ONEFLOW_CORE_KERNEL_CONV_BIAS_GRAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CONV_BIAS_GRAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ConvBiasGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvBiasGradKernel);
  ConvBiasGradKernel() = default;
  ~ConvBiasGradKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  const PbMessage& GetCustomizedOpConf() const override;
};

template<DeviceType device_type, typename T>
struct ConvBiasGradKernelUtil final {
  static void Compute(DeviceCtx* ctx, const std::string& format, const Blob* dy, Blob* bias_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONV_BIAS_GRAD_KERNEL_H_
