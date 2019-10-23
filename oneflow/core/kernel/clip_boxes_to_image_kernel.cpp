#include "oneflow/core/kernel/clip_boxes_to_image_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ClipBoxesToImageKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ClipBoxesToImageKernel);
  ClipBoxesToImageKernel() = default;
  ~ClipBoxesToImageKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* boxes_blob = BnInOp2Blob("boxes");
    const Blob* image_size_blob = BnInOp2Blob("image_size");
    CHECK_EQ(boxes_blob->shape().elem_cnt() % 4, 0);
    CHECK_EQ(image_size_blob->shape().NumAxes(), 1);
    CHECK_EQ(image_size_blob->shape().At(0), 2);
    ClipBoxesToImageUtil<device_type, T>::ClipBoxes(
        ctx.device_ctx, boxes_blob->shape().elem_cnt() / 4, boxes_blob->dptr<T>(),
        image_size_blob->dptr<int32_t>(), BnInOp2Blob("out")->mut_dptr<T>());
  }
};

template<typename T>
struct ClipBoxesToImageUtil<DeviceType::kCPU, T> {
  static void ClipBoxes(DeviceCtx* ctx, const int32_t num_boxes, const T* boxes_ptr,
                        const int32_t* image_size_ptr, T* out_ptr) {
    const int32_t image_height = image_size_ptr[0];
    const int32_t image_width = image_size_ptr[1];
    const T TO_REMOVE = 1;
    FOR_RANGE(int32_t, i, 0, num_boxes) {
      out_ptr[i * 4] = std::min(std::max(boxes_ptr[i * 4], GetZeroVal<T>()),
                                static_cast<T>(image_width) - TO_REMOVE);
      out_ptr[i * 4 + 1] = std::min(std::max(boxes_ptr[i * 4 + 1], GetZeroVal<T>()),
                                    static_cast<T>(image_height) - TO_REMOVE);
      out_ptr[i * 4 + 2] = std::min(std::max(boxes_ptr[i * 4 + 2], GetZeroVal<T>()),
                                    static_cast<T>(image_width) - TO_REMOVE);
      out_ptr[i * 4 + 3] = std::min(std::max(boxes_ptr[i * 4 + 3], GetZeroVal<T>()),
                                    static_cast<T>(image_height) - TO_REMOVE);
    }
  }
};

#define REGISTER_CLIP_BOXES_TO_IMAGE_KERNEL(dev, dtype)                                  \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kClipBoxesToImageConf, dev, dtype, \
                                        ClipBoxesToImageKernel<dev, dtype>)

REGISTER_CLIP_BOXES_TO_IMAGE_KERNEL(DeviceType::kGPU, float);
REGISTER_CLIP_BOXES_TO_IMAGE_KERNEL(DeviceType::kGPU, double);
REGISTER_CLIP_BOXES_TO_IMAGE_KERNEL(DeviceType::kCPU, float);
REGISTER_CLIP_BOXES_TO_IMAGE_KERNEL(DeviceType::kCPU, double);

}  // namespace oneflow
