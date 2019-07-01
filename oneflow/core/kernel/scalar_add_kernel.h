#ifndef ONEFLOW_CORE_KERNEL_SCALAR_ADD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SCALAR_ADD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ScalarAddKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScalarAddKernel);
  ScalarAddKernel() = default;
  ~ScalarAddKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  const PbMessage& GetCustomizedOpConf() const override {
    return this->op_conf().scalar_add_conf();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SCALAR_ADD_KERNEL_H_
