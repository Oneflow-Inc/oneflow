#ifndef ONEFLOW_CORE_KERNEL_REDUCE_IDENTITY_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_REDUCE_IDENTITY_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class ReduceIdentityKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceIdentityKernel);
  ReduceIdentityKernel() = default;
  ~ReduceIdentityKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  const PbMessage& GetCustomizedOpConf() const override {
    return this->op_conf().reduce_identity_conf();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_REDUCE_IDENTITY_KERNEL_H_
