#ifndef ONEFLOW_CORE_KERNEL_COPY_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_COPY_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

#ifdef WITH_CUDA

class CopyKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyKernel);
  CopyKernel() = default;
  ~CopyKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

#endif

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_COPY_KERNEL_H_
