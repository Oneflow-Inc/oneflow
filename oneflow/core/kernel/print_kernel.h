#ifndef ONEFLOW_CORE_KERNEL_PRINT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_PRINT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class PrintKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PrintKernel);
  PrintKernel() = default;
  ~PrintKernel() = default;

 private:
  void VirtualKernelInit() override;
  void Forward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;

  std::unique_ptr<PersistentOutStream> out_stream_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PRINT_KERNEL_H_
