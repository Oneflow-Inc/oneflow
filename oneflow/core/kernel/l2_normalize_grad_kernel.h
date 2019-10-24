#ifndef ONEFLOW_CORE_KERNEL_L2_NORMALIZE_GRAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_L2_NORMALIZE_GRAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class L2NormalizeGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(L2NormalizeGradKernel);
  L2NormalizeGradKernel() = default;
  ~L2NormalizeGradKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_L2_NORMALIZE_GRAD_KERNEL_H_
