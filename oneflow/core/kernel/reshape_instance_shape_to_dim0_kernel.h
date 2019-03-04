#ifndef ONEFLOW_CORE_KERNEL_RESHAPE_INSTANCE_SHAPE_TO_DIM0_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RESHAPE_INSTANCE_SHAPE_TO_DIM0_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type>
class ReshapeInstanceShapeToDim0Kernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReshapeInstanceShapeToDim0Kernel);
  ReshapeInstanceShapeToDim0Kernel() = default;
  ~ReshapeInstanceShapeToDim0Kernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;

  void ForwardDim0ValidNum(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardInstanceShape(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BackwardInDiffDim0ValidNum(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BackwardInstanceShape(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RESHAPE_INSTANCE_SHAPE_TO_DIM0_KERNEL_H_
