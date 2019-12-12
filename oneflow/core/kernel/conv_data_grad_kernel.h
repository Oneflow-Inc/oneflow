#ifndef ONEFLOW_CORE_KERNEL_CONV_DATA_GRAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CONV_DATA_GRAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ConvDataGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvDataGradKernel);
  ConvDataGradKernel() = default;
  ~ConvDataGradKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  const PbMessage& GetCustomizedOpConf() const override;
};

template<DeviceType device_type, typename T>
struct ConvDataGradKernelUtil final {
  static void Compute(DeviceCtx* ctx, const ConvDataGradKernelConf& kernel_conf,
                      const ConvConf& conf, const Blob* dy, const Blob* filter, Blob* dx, Blob* buf,
                      const bool enable_true_half);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONV_DATA_GRAD_KERNEL_H_
