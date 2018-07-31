#ifndef ONEFLOW_CORE_KERNEL_RESULTS_WITH_NMS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RESULTS_WITH_NMS_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename T>
class ResultsWithNmsKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ResultsWithNmsKernel);
  ResultsWithNmsKernel() = default;
  ~ResultsWithNmsKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RESULTS_WITH_NMS_KERNEL_H_
