#ifndef ONEFLOW_CORE_KERNEL_ROI_ALIGN_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_ROI_ALIGN_KERNEL_H_

#include "oneflow/core/kernel/roi_resize_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class RoIAlignKernel final : public RoIResizeKernel<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RoIAlignKernel);
  RoIAlignKernel() = default;
  ~RoIAlignKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BackwardInDiffDim0ValidNum(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardRecordIdInDevicePiece(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardInstanceShape(const KernelCtx& ctx,
                            std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
struct RoIAlignKernelUtil {
  static void Forward(const KernelCtx& ctx, const RoIAlignOpConf& conf, const Blob* in_blob,
                      const Blob* rois_blob, Blob* out_blob);
  static void Backward(const KernelCtx& ctx, const RoIAlignOpConf& conf, const Blob* out_diff_blob,
                       const Blob* rois_blob, Blob* in_diff_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ROI_ALIGN_KERNEL_H_
