#ifndef ONEFLOW_CORE_KERNEL_COPY_LOCAL_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_COPY_LOCAL_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

#ifdef WITH_CUDA

class CopyLocalKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyLocalKernel);
  CopyLocalKernel() = default;
  ~CopyLocalKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

#endif

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_COPY_LOCAL_KERNEL_H_
