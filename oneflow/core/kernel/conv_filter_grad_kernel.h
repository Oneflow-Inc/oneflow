#ifndef ONEFLOW_CORE_KERNEL_CONV_FILTER_GRAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CONV_FILTER_GRAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ConvFilterGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvFilterGradKernel);
  ConvFilterGradKernel() = default;
  ~ConvFilterGradKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  const PbMessage& GetCustomizedOpConf() const override;
};

template<DeviceType device_type, typename T>
struct ConvFilterGradKernelUtil final {
  static void Compute(DeviceCtx* ctx, const ConvFilterGradKernelConf& kernel_conf,
                      const ConvConf& conf, const Blob* x, const Blob* dy, Blob* filter_diff,
                      Blob* buf, const bool enable_true_half);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONV_FILTER_GRAD_KERNEL_H_
