#ifndef ONEFLOW_CORE_KERNEL_INSTANCE_STACK_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_INSTANCE_STACK_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class InstanceStackKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InstanceStackKernel);
  InstanceStackKernel() = default;
  ~InstanceStackKernel() = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BackwardDataContent(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardInstanceShape(const KernelCtx& ctx,
                            std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDim1ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDim2ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BackwardInstanceShape(const KernelCtx& ctx,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  mutable bool is_first_in_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_INSTANCE_STACK_KERNEL_H_
