#ifndef ONEFLOW_CORE_KERNEL_CAST_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CAST_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class CastKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CastKernel);
  CastKernel() = default;
  ~CastKernel() = default;

 private:
  bool HasSameShapeBetweenInOut() const override { return true; }
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override {
    TODO();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CAST_KERNEL_H_
