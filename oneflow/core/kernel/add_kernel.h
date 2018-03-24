#ifndef ONEFLOW_CORE_KERNEL_ADD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_ADD_KERNEL_H_

#include "oneflow/core/kernel/cwise_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class AddKernel final : public CWiseKernel<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddKernel);
  AddKernel() = default;
  ~AddKernel() = default;

 private:
  void ForwardDataContent(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void BackwardDataContent(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ADD_KERNEL_H_
