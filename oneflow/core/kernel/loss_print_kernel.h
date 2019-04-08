#ifndef ONEFLOW_CORE_KERNEL_LOSS_PRINT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LOSS_PRINT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename T>
class LossPrintKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossPrintKernel);
  LossPrintKernel() = default;
  ~LossPrintKernel() = default;

  void Forward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;

 private:
  T GetReductionCoefficient(T weight_scalar, ScalarReductionType reduction, int32_t n) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LOSS_PRINT_KERNEL_H_
