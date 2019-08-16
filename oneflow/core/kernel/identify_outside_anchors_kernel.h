#ifndef ONEFLOW_CORE_KERNEL_IDENTIFY_OUTSIDE_ANCHORS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_IDENTIFY_OUTSIDE_ANCHORS_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class IdentifyOutsideAnchorsKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IdentifyOutsideAnchorsKernel);
  IdentifyOutsideAnchorsKernel() = default;
  ~IdentifyOutsideAnchorsKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct IdentifyOutsideAnchorsUtil final {
  static void Forward(DeviceCtx* ctx, const Blob* anchors_blob, const Blob* image_size_blob,
                      Blob* identification_blob, float tolerance);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_IDENTIFY_OUTSIDE_ANCHORS_KERNEL_H_
