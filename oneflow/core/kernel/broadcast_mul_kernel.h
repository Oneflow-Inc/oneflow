#ifndef ONEFLOW_CORE_KERNEL_BROADCAST_MUL_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BROADCAST_MUL_KERNEL_H_

#include "oneflow/core/kernel/broadcast_binary_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BroadcastMulKernel final : public BroadcastBinaryKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastMulKernel);
  BroadcastMulKernel() = default;
  ~BroadcastMulKernel() = default;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BROADCAST_MUL_KERNEL_H_
