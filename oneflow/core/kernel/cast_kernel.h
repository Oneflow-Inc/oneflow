#ifndef ONEFLOW_CORE_KERNEL_CAST_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CAST_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class CastKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CastKernel);
  CastKernel() = default;
  ~CastKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardLoD(const KernelCtx& ctx,
                  std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CAST_KERNEL_H_
