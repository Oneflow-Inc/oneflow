#include "oneflow/core/kernel/clip_boxes_to_image_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ClipBoxesToImageKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

template<DeviceType device_type, typename T>
void ClipBoxesToImageKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx&, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

template<typename T>
struct ClipBoxesToImageUtil<DeviceType::kCPU, T> {
  static void Forward() {
    // TODO
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kClipBoxesToImageConf, ClipBoxesToImageKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
