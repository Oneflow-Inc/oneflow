#ifndef ONEFLOW_CORE_KERNEL_BROADCAST_BINARY_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BROADCAST_BINARY_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BroadcastBinaryKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastBinaryKernel);
  BroadcastBinaryKernel() = default;
  virtual ~BroadcastBinaryKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override {
    // TODO;
  }
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override {
    // TODO;
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BROADCAST_BINARY_KERNEL_H_
