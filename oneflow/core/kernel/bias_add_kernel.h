#ifndef ONEFLOW_CORE_KERNEL_BIAS_ADD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BIAS_ADD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BiasAddKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BiasAddKernel);
  BiasAddKernel() = default;
  ~BiasAddKernel() = default;

 private:
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void InitConstBufBlobs(DeviceCtx*,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  const PbMessage& GetCustomizedOpConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BIAS_ADD_KERNEL_H_
