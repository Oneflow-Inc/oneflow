#ifndef ONEFLOW_CORE_KERNEL_BROADCAST_SUB_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BROADCAST_SUB_KERNEL_H_

#include "oneflow/core/kernel/broadcast_binary_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BroadcastSubKernel final : public BroadcastBinaryKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastSubKernel);
  BroadcastSubKernel() = default;
  ~BroadcastSubKernel() = default;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BROADCAST_SUB_KERNEL_H_
