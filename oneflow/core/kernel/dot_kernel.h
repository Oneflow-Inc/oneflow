#ifndef ONEFLOW_CORE_KERNEL_DOT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_DOT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
namespace oneflow {

template<DeviceType device_type, typename T>
class DotKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DotKernel);
  DotKernel() = default;
  ~DotKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
  void InitConstBufBlobs(DeviceCtx*,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  const PbMessage& GetCustomizedOpConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOE_CORE_KERNEL_DOT_KERNEL_H_
