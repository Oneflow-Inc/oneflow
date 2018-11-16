#ifndef ONEFLOW_CORE_KERNEL_CONSTANT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CONSTANT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ConstantKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConstantKernel);
  ConstantKernel() : is_init_(false) {}
  ~ConstantKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  const PbMessage& GetCustomizedOpConf() const override;

  mutable bool is_init_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONSTANT_KERNEL_H_
