#ifndef ONEFLOW_CORE_KERNEL_PRELU_DATA_GRAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_PRELU_DATA_GRAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class PReluDataGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PReluDataGradKernel);
  PReluDataGradKernel() = default;
  ~PReluDataGradKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct PReluDataGradKernelUtil {
  static void Compute(const KernelCtx& ctx, const PReluDataGradOpConf& conf, const Blob* x_blob,
                      const Blob* alpha_blob, const Blob* dy_blob, Blob* dx_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PRELU_DATA_GRAD_KERNEL_H_
