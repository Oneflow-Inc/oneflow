#ifndef ONEFLOW_CORE_KERNEL_IDENTITY_LOSS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_IDENTITY_LOSS_KERNEL_H_

#include "oneflow/core/kernel/loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType>
class IdentityLossKernel final : public LossKernel<device_type, PredType> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IdentityLossKernel);
  IdentityLossKernel() = default;
  ~IdentityLossKernel() = default;

 private:
  void VirtualLossForwardDataContent(const KernelCtx&,
                                     std::function<Blob*(const std::string&)>) const override;
  const LossKernelConf& GetLossKernelConf(const KernelConf& kernel_conf) const override;
  void InitConstBufBlobs(DeviceCtx* ctx,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_IDENTITY_LOSS_KERNEL_H_
