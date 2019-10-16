#ifndef ONEFLOW_CORE_KERNEL_FOREIGN_WATCH_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_FOREIGN_WATCH_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class ForeignWatchKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForeignWatchKernel);
  ForeignWatchKernel() = default;
  ~ForeignWatchKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_FOREIGN_WATCH_KERNEL_H_
