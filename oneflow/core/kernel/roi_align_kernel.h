#ifndef ONEFLOW_CORE_KERNEL_ROI_ALIGN_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_ROI_ALIGN_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct RoiAlignKernelUtil {
  static void Forward(const KernelCtx& ctx, const RoiAlignConf& conf, const Blob* in_blob,
                      const Blob* rois_blob, Blob* out_blob);
  static void Backward(const KernelCtx& ctx, const RoiAlignConf& conf, const Blob* out_diff_blob,
                       const Blob* rois_blob, Blob* in_diff_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ROI_ALIGN_KERNEL_H_
