#ifndef ONEFLOW_CORE_KERNEL_MEAN_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MEAN_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MeanKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MeanKernel);
  MeanKernel() = default;
  ~MeanKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
  const PbMessage& GetCustomizedOpConf() const override { return this->op_conf().mean_conf(); }
};

}  // namespace oneflow

#endif  // ONEFLOE_CORE_KERNEL_MEAN_KERNEL_H_
