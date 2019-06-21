#ifndef ONEFLOW_CORE_KERNEL_REDUCE_MEAN_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_REDUCE_MEAN_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ReduceMeanKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceMeanKernel);
  ReduceMeanKernel() = default;
  ~ReduceMeanKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_REDUCE_MEAN_KERNEL_H_
