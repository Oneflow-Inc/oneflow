#ifndef ONEFLOW_CORE_KERNEL_BROADCAST_DIV_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BROADCAST_DIV_KERNEL_H_

#include "oneflow/core/kernel/broadcast_binary_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BroadcastDivKernel final : public BroadcastBinaryKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastDivKernel);
  BroadcastDivKernel() = default;
  ~BroadcastDivKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BROADCAST_DIV_KERNEL_H_
