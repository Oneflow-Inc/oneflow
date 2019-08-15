#ifndef ONEFLOW_CORE_KERNEL_BROADCAST_NOT_EQUAL_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BROADCAST_NOT_EQUAL_KERNEL_H_

#include "oneflow/core/kernel/broadcast_binary_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BroadcastNotEqualKernel final : public BroadcastBinaryKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastNotEqualKernel);
  BroadcastNotEqualKernel() = default;
  ~BroadcastNotEqualKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BROADCAST_NOT_EQUAL_KERNEL_H_
