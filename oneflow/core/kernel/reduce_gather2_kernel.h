#ifndef ONEFLOW_CORE_KERNEL_REDUCE_GATHER2_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_REDUCE_GATHER2_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class ReduceGather2Kernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceGather2Kernel);
  ReduceGather2Kernel() = default;
  ~ReduceGather2Kernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_REDUCE_GATHER2_KERNEL_H_
