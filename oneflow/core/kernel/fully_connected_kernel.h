#ifndef ONEFLOW_CORE_KERNEL_FULLY_CONNECTED_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_FULLY_CONNECTED_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class FullyConnectedKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FullyConnectedKernel);
  FullyConnectedKernel() = default;
  ~FullyConnectedKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void InitConstBufBlobs(DeviceCtx*,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  const PbMessage& GetCustomizedOpConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_FULLY_CONNECTED_KERNEL_H_
