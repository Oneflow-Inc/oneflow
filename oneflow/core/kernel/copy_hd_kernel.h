#ifndef ONEFLOW_CORE_KERNEL_COPY_HD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_COPY_HD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

#ifdef WITH_CUDA

class CopyHdKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdKernel);
  CopyHdKernel() = default;
  ~CopyHdKernel() = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  void Forward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;

  cudaMemcpyKind cp_kind_;
};

#endif

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_COPY_HD_KERNEL_H_
