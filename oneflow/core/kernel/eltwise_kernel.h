#ifndef ONEFLOW_CORE_KERNEL_ELTWISE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_ELTWISE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class EltwiseKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EltwiseKernel);
  EltwiseKernel() = default;
  ~EltwiseKernel() = default;

 private:
  void ForwardDataContent(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void ForwardDataId(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void BackwardDataContent(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ELTWISE_KERNEL_H_
