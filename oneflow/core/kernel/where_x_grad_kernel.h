#ifndef ONEFLOW_CORE_KERNEL_WHERE_X_GRAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_WHERE_X_GRAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class WhereXGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WhereXGradKernel);
  WhereXGradKernel() = default;
  ~WhereXGradKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_WHERE_X_GRAD_KERNEL_H_
