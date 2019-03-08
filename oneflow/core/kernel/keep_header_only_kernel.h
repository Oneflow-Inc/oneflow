#ifndef ONEFLOW_CORE_KERNEL_KEEP_HEADER_ONLY_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_KEEP_HEADER_ONLY_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type>
class KeepHeaderOnlyKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KeepHeaderOnlyKernel);
  KeepHeaderOnlyKernel() = default;
  ~KeepHeaderOnlyKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override {}
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KEEP_HEADER_ONLY_KERNEL_H_
