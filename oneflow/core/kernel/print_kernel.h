#ifndef ONEFLOW_CORE_KERNEL_PRINT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_PRINT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class PrintKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PrintKernel);
  PrintKernel() = default;
  ~PrintKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PRINT_KERNEL_H_
