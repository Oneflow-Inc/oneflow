#ifndef ONEFLOW_CORE_KERNEL_IMAGE_SEGMENTATION_MASK_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_IMAGE_SEGMENTATION_MASK_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename T>
class ImageSegmentationMaskKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ImageSegmentationMaskKernel);
  ImageSegmentationMaskKernel() = default;
  ~ImageSegmentationMaskKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardRecordIdInDevicePiece(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_IMAGE_SEGMENTATION_MASK_KERNEL_H_
