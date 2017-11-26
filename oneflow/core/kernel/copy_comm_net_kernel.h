#ifndef ONEFLOW_CORE_KERNEL_COPY_COMM_NET_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_COPY_COMM_NET_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class CopyCommNetKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyCommNetKernel);
  CopyCommNetKernel() = default;
  ~CopyCommNetKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override {
    UNEXPECTED_RUN();
  }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_COPY_COMM_NET_KERNEL_H_
