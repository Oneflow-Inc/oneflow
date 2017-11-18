#ifndef ONEFLOW_CORE_KERNEL_LOSS_RECORD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LOSS_RECORD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename T>
class LossRecordKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossRecordKernel);
  LossRecordKernel() = default;
  ~LossRecordKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LOSS_RECORD_KERNEL_H_
