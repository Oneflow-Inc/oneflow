#ifndef ONEFLOW_CORE_KERNEL_RLE_SEGMENTATION_RESULT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RLE_SEGMENTATION_RESULT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename T>
class RleSegmentationResultKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RleSegmentationResultKernel);
  RleSegmentationResultKernel() = default;
  ~RleSegmentationResultKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RLE_SEGMENTATION_RESULT_KERNEL_H_
