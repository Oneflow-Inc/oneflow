#ifndef ONEFLOW_CORE_KERNEL_BATCH_GATHER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BATCH_GATHER_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class BatchGatherKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BatchGatherKernel);
  BatchGatherKernel() = default;
  ~BatchGatherKernel() override = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BATCH_GATHER_KERNEL_H_
