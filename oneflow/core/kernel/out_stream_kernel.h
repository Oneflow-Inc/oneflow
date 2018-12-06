#ifndef ONEFLOW_CORE_KERNEL_OUT_STREAM_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_OUT_STREAM_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class OutStreamKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OutStreamKernel);
  OutStreamKernel() = default;
  ~OutStreamKernel() = default;

 private:
  void Forward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_OUT_STREAM_KERNEL_H_