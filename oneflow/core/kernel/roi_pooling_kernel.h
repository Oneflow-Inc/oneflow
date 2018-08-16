#ifndef ONEFLOW_CORE_KERNEL_ROI_POOLING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_ROI_POOLING_KERNEL_H_

#include "oneflow/core/kernel/roi_resize_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class RoIPoolingKernel final : public RoIResizeKernel<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RoIPoolingKernel);
  RoIPoolingKernel() = default;
  ~RoIPoolingKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
class RoIPoolingKernelUtil {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RoIPoolingKernelUtil);
  RoIPoolingKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const RoIPoolingOpConf& conf, const Blob* in_blob,
                      const Blob* rois_blob, Blob* out_blob, Blob* idx_blob);
  static void Backward(const KernelCtx& ctx, const RoIPoolingOpConf& conf,
                       const Blob* out_diff_blob, const Blob* rois_blob, const Blob* idx_blob,
                       Blob* in_diff_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ROI_POOLING_KERNEL_H_
