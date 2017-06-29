#ifndef ONEFLOW_CORE_KERNEL_INNERPRODUCT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_INNERPRODUCT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
class InnerProductKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InnerProductKernel);
  InnerProductKernel() = default;
  ~InnerProductKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_INNERPRODUCT_KERNEL_H_
