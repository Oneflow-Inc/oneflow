#ifndef ONEFLOW_CORE_OPERATOR_CLIP_BOXES_TO_IMAGE_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_CLIP_BOXES_TO_IMAGE_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ClipBoxesToImageKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ClipBoxesToImageKernel);
  ClipBoxesToImageKernel() = default;
  ~ClipBoxesToImageKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct ClipBoxesToImageUtil {
  static void Forward(DeviceCtx* ctx, const int32_t num_boxes, const T* boxes_ptr,
                      const int32_t* image_size_ptr, T* out_ptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CLIP_BOXES_TO_IMAGE_KERNEL_OP_H_
