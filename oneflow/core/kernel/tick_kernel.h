#ifndef ONEFLOW_CORE_KERNEL_TICK_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_TICK_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class TickKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TickKernel);
  TickKernel() = default;
  ~TickKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {}
  void BackwardDataContent(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    UNIMPLEMENTED();
  }
  const PbMessage& GetCustomizedOpConf() const override { return this->op_conf().tick_conf(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_TICK_KERNEL_H_
