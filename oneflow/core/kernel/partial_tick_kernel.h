#ifndef ONEFLOW_CORE_KERNEL_PARTIAL_TICK_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_PARTIAL_TICK_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class PartialTickKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PartialTickKernel);
  PartialTickKernel() = default;
  ~PartialTickKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {}
  const PbMessage& GetCustomizedOpConf() const override { return this->op_conf().tick_conf(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PARTIAL_TICK_KERNEL_H_
