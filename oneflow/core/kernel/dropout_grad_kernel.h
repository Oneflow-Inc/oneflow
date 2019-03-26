#ifndef ONEFLOW_CORE_KERNEL_DROPOUT_GRAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_DROPOUT_GRAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/kernel/random_generator.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class DropoutGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DropoutGradKernel);
  DropoutGradKernel() = default;
  ~DropoutGradKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void DropoutBackward(DeviceCtx* ctx, const int64_t n, float dropout_rate, const T* dy,
                       const float* random_mask, T* dx) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DROPOUT_GRAD_KERNEL_H_
