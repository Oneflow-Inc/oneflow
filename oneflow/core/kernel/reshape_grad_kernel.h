#ifndef ONEFLOW_CORE_KERNEL_RESHAPE_GRAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RESHAPE_GRAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type>
class ReshapeGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReshapeGradKernel);
  ReshapeGradKernel() = default;
  ~ReshapeGradKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RESHAPE_GRAD_KERNEL_H_
