#ifndef ONEFLOW_CORE_KERNEL_TUPLE_IDENTITY_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_TUPLE_IDENTITY_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type>
class TupleIdentityKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TupleIdentityKernel);
  TupleIdentityKernel() = default;
  ~TupleIdentityKernel() = default;

 private:
  void ForwardHeader(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_TUPLE_IDENTITY_KERNEL_H_
