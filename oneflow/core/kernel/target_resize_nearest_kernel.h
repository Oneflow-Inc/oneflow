#ifndef ONEFLOW_CORE_KERNEL_TARGET_RESIZE_NEAREST_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_TARGET_RESIZE_NEAREST_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class TargetResizeNearestKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TargetResizeNearestKernel);
  TargetResizeNearestKernel() = default;
  ~TargetResizeNearestKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
  void ForwardInstanceShape(const KernelCtx& ctx,
                            std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_TARGET_RESIZE_NEAREST_KERNEL_H_
