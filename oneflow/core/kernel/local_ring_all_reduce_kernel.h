#ifndef ONEFLOW_CORE_KERNEL_LOCAL_RING_ALL_REDUCE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LOCAL_RING_ALL_REDUCE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/register/tensor_slice_copier.h"
#include "oneflow/core/device/memory_copier.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class LocalRingAllReduceKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalRingAllReduceKernel);
  LocalRingAllReduceKernel() = default;
  ~LocalRingAllReduceKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LOCAL_RING_ALL_REDUCE_KERNEL_H_
