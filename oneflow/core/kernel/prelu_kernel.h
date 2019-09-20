#ifndef ONEFLOW_CORE_KERNEL_PRELU_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_PRELU_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class PReluKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PReluKernel);
  PReluKernel() = default;
  ~PReluKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct PReluKernelUtil {
  static void Forward(const KernelCtx& ctx, const PReluOpConf& conf, const Blob* in_blob,
                      const Blob* alpha_blob, Blob* out_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PRELU_KERNEL_H_
