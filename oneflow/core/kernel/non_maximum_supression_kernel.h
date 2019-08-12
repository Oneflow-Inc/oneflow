#ifndef ONEFLOW_CORE_OPERATOR_NON_MAXIMUM_SUPRESSION_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_NON_MAXIMUM_SUPRESSION_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class NonMaximumSupressionKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NonMaximumSupressionKernel);
  NonMaximumSupressionKernel() = default;
  ~NonMaximumSupressionKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct NonMaximumSupressionUtil {
  static void Forward();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NON_MAXIMUM_SUPRESSION_KERNEL_OP_H_
