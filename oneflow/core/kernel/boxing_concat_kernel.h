#ifndef ONEFLOW_CORE_KERNEL_BOXING_CONCAT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BOXING_CONCAT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class BoxingConcatKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingConcatKernel);
  BoxingConcatKernel() = default;
  ~BoxingConcatKernel() = default;

 private:
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BOXING_CONCAT_KERNEL_H_
