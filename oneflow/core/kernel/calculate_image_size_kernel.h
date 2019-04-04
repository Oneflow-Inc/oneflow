#ifndef ONEFLOW_CORE_KERNEL_CALCULATE_IMAGE_SIZE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CALCULATE_IMAGE_SIZE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<typename T>
class CalculateImageSizeKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CalculateImageSizeKernel);
  CalculateImageSizeKernel() = default;
  ~CalculateImageSizeKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CALCULATE_IMAGE_SIZE_KERNEL_H_
