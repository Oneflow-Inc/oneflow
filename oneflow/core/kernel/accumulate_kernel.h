#ifndef ONEFLOW_CORE_KERNEL_ACCUMULATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_ACCUMULATE_KERNEL_H_

#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class AccumulateKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccumulateKernel);
  AccumulateKernel() = default;
  ~AccumulateKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ACCUMULATE_KERNEL_H_
