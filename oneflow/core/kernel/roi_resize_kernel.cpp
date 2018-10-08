#include "oneflow/core/kernel/roi_resize_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void RoIResizeKernel<device_type>::ForwardDataId(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyDataIdFrom(ctx.device_ctx, BnInOp2Blob("rois"));
}

template<DeviceType device_type>
void RoIResizeKernel<device_type>::ForwardColNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyColNumFrom(ctx.device_ctx, BnInOp2Blob("rois"));
}

template class RoIResizeKernel<DeviceType::kCPU>;
template class RoIResizeKernel<DeviceType::kGPU>;

}  // namespace oneflow
