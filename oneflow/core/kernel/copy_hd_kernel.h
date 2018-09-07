#ifndef ONEFLOW_CORE_KERNEL_COPY_HD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_COPY_HD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

#ifdef WITH_CUDA

class CopyHdKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdKernel);
  CopyHdKernel() : act_cnt_(0) {}
  ~CopyHdKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  mutable int64_t act_cnt_;
};

#endif

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_COPY_HD_KERNEL_H_
