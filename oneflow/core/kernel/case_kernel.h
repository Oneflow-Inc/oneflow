#ifndef ONEFLOW_CORE_KERNEL_CASE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CASE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename T>
class CaseKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CaseKernel);
  CaseKernel() = default;
  ~CaseKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CASE_KERNEL_H_
