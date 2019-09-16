#ifndef ONEFLOW_CORE_KERNEL_FOREIGN_OUTPUT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_FOREIGN_OUTPUT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class ForeignOutputKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForeignOutputKernel);
  ForeignOutputKernel() = default;
  ~ForeignOutputKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_FOREIGN_OUTPUT_KERNEL_H_
