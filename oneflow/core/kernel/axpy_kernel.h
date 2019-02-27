#ifndef ONEFLOW_CORE_KERNEL_AXPY_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_AXPY_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class AxpyKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AxpyKernel);
  AxpyKernel() = default;
  ~AxpyKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  const PbMessage& GetCustomizedOpConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ADD_ASSIGN_KERNEL_H_
