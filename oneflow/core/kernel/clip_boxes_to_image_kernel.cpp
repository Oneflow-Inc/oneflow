#include "oneflow/core/kernel/clip_boxes_to_image_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ClipBoxesToImageKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* boxes_blob = BnInOp2Blob("boxes");
  const Blob* image_size_blob = BnInOp2Blob("image_size");
  CHECK_EQ(boxes_blob->shape().NumAxes(), 2);
  CHECK_EQ(boxes_blob->shape().At(1), 4);
  CHECK(!boxes_blob->has_instance_shape_field());
  CHECK_EQ(image_size_blob->shape().NumAxes(), 1);
  CHECK_EQ(image_size_blob->shape().At(0), 2);
  ClipBoxesToImageUtil<device_type, T>::Forward(
      ctx.device_ctx, boxes_blob->shape().At(0), boxes_blob->dptr<T>(),
      image_size_blob->dptr<int32_t>(), BnInOp2Blob("out")->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void ClipBoxesToImageKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyDim0ValidNumFrom(ctx.device_ctx, BnInOp2Blob("boxes"));
}

template<typename T>
struct ClipBoxesToImageUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const int32_t num_boxes, const T* boxes_ptr,
                      const int32_t* image_size_ptr, T* out_ptr) {
    const int32_t image_height = image_size_ptr[0];
    const int32_t image_width = image_size_ptr[1];
    const T TO_REMOVE = 1;
    FOR_RANGE(int32_t, i, 0, num_boxes) {
      out_ptr[i * 4] = std::min(std::max(boxes_ptr[i * 4], ZeroVal<T>::value),
                                static_cast<T>(image_width) - TO_REMOVE);
      out_ptr[i * 4 + 1] = std::min(std::max(boxes_ptr[i * 4 + 1], ZeroVal<T>::value),
                                    static_cast<T>(image_height) - TO_REMOVE);
      out_ptr[i * 4 + 2] = std::min(std::max(boxes_ptr[i * 4 + 2], ZeroVal<T>::value),
                                    static_cast<T>(image_width) - TO_REMOVE);
      out_ptr[i * 4 + 3] = std::min(std::max(boxes_ptr[i * 4 + 3], ZeroVal<T>::value),
                                    static_cast<T>(image_height) - TO_REMOVE);
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kClipBoxesToImageConf, ClipBoxesToImageKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
