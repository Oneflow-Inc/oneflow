#ifndef ONEFLOW_CORE_KERNEL_GATHER_MS0_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_GATHER_MS0_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class GatherMs0Kernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherMs0Kernel);
  GatherMs0Kernel() = default;
  ~GatherMs0Kernel() override = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_GATHER_MS0_KERNEL_H_
