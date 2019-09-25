#ifndef ONEFLOW_CORE_KERNEL_UPSAMPLE_NEAREST_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_UPSAMPLE_NEAREST_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class UpsampleNearestKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UpsampleNearestKernel);
  UpsampleNearestKernel() = default;
  ~UpsampleNearestKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
class UpsampleNearestGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UpsampleNearestGradKernel);
  UpsampleNearestGradKernel() = default;
  ~UpsampleNearestGradKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct UpsampleNearestUtil {
  static void Forward(const KernelCtx& ctx, const float scale_h, const float scale_w,
                      const bool align_corners, const Blob* in_blob, Blob* out_blob);
  static void Backward(const KernelCtx& ctx, const float scale_h, const float scale_w,
                       const bool align_corners, const Blob* dy_blob, Blob* dx_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_UPSAMPLE_NEAREST_KERNEL_H_
