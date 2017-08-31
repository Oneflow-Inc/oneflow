#ifndef ONEFLOW_CORE_KERNEL_CLONE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CLONE_KERNEL_H_

#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class CloneKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CloneKernel);
  CloneKernel() = default;
  ~CloneKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CLONE_KERNEL_H_
