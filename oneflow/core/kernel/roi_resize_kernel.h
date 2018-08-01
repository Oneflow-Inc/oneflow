#ifndef ONEFLOW_CORE_KERNEL_ROI_RESIZE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_ROI_RESIZE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class RoIResizeKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RoIResizeKernel);
  RoIResizeKernel() = default;
  virtual ~RoIResizeKernel() = default;

 private:
  void ForwardDataId(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
  void ForwardColNum(const KernelCtx& ctx, std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ROI_RESIZE_KERNEL_H_
