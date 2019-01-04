#ifndef ONEFLOW_CORE_KERNEL_NCCL_INTER_DEVICE_REDUCE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_NCCL_INTER_DEVICE_REDUCE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename T>
class NcclInterDeviceReduceKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclInterDeviceReduceKernel);
  NcclInterDeviceReduceKernel() = default;
  ~NcclInterDeviceReduceKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BackwardDataContent(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  const PbMessage& GetCustomizedOpConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NCCL_INTER_DEVICE_REDUCE_KERNEL_H_
