#ifndef ONEFLOW_CORE_KERNEL_UNPACK_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_UNPACK_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class UnpackKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnpackKernel);
  UnpackKernel() = default;
  ~UnpackKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_UNPACK_KERNEL_H_
