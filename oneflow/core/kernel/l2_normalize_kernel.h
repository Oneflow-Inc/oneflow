#ifndef ONEFLOW_CORE_KERNEL_L2_NORMALIZE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_L2_NORMALIZE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class L2NormalizeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(L2NormalizeKernel);
  L2NormalizeKernel() = default;
  ~L2NormalizeKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_L2_NORMALIZE_KERNEL_H_
