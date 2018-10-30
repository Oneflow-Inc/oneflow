#ifndef ONEFLOW_CORE_KERNEL_LOG_COUNTER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LOG_COUNTER_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class LogCounterKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogCounterKernel);
  LogCounterKernel() = default;
  ~LogCounterKernel() = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  void Forward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;

  std::unique_ptr<int64_t> counter_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LOG_COUNTER_KERNEL_H_
