#ifndef ONEFLOW_CORE_KERNEL_REPEAT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_REPEAT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class RepeatKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RepeatKernel);
  RepeatKernel() = default;
  ~RepeatKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
  void BackwardInDiffLossInstanceNum(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    UNIMPLEMENTED();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_REPEAT_KERNEL_H_
