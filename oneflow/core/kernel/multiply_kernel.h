#ifndef ONEFLOW_CORE_KERNEL_MULTIPLY_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MULTIPLY_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
namespace oneflow {

template<DeviceType device_type, typename T>
class MultiplyKernel final : public KernelIfWithModel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultiplyKernel);
  MultiplyKernel() = default;
  ~MultiplyKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  const PbMessage& GetCustomizedOpConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOE_CORE_KERNEL_MULTIPLY_KERNEL_H_
