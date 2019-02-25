#ifndef ONEFLOW_CORE_KERNEL_HALF_TO_FLOAT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_HALF_TO_FLOAT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class HalfToFloatKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HalfToFloatKernel);
  HalfToFloatKernel() = default;
  ~HalfToFloatKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BackwardDataContent(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  const PbMessage& GetCustomizedOpConf() const override {
    return this->op_conf().half_to_float_conf();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_HALF_TO_FLOAT_KERNEL_H_
