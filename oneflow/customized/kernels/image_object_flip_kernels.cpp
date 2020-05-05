#include "oneflow/core/framework/framework.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/customized/image/image_util.h"
#include <opencv2/opencv.hpp>

namespace oneflow {

namespace {

enum class FlipCode : int {
  kNonFlip = 0,
  kHorizontalFlip = 1,
  kVerticalFlip = 2,
  kBothDirectionFlip = 3,
};

template<typename T>
void FlipBoxes(TensorBuffer* boxes_buffer, int32_t image_height, int32_t image_width,
               int flip_code) {
  int num_boxes = boxes_buffer->shape().At(0);
  FOR_RANGE(int, i, 0, num_boxes) {
    T* cur_box_ptr = boxes_buffer->mut_data<T>() + i * 4;
    if (flip_code & (int)FlipCode::kHorizontalFlip) {
      T xmin = cur_box_ptr[0];
      T xmax = cur_box_ptr[2];
      cur_box_ptr[0] = image_width - xmax - static_cast<T>(1);
      cur_box_ptr[2] = image_width - xmin - static_cast<T>(1);
    }
    if (flip_code & (int)FlipCode::kVerticalFlip) {
      T ymin = cur_box_ptr[1];
      T ymax = cur_box_ptr[3];
      cur_box_ptr[1] = image_height - ymax - static_cast<T>(1);
      cur_box_ptr[3] = image_height - ymin - static_cast<T>(1);
    }
  }
}

#define MAKE_FLIP_BOXES_SWITCH_ENTRY(func_name, T) func_name<T>
DEFINE_STATIC_SWITCH_FUNC(void, FlipBoxes, MAKE_FLIP_BOXES_SWITCH_ENTRY,
                          MAKE_DATA_TYPE_CTRV_SEQ(FLOATING_DATA_TYPE_SEQ));

#undef MAKE_FLIP_BOXES_SWITCH_ENTRY

}  // namespace

class ObjectBboxFlipKernel final : public user_op::OpKernel {
 public:
  ObjectBboxFlipKernel() = default;
  ~ObjectBboxFlipKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* bbox_tensor = ctx->Tensor4ArgNameAndIndex("bbox", 0);
    const user_op::Tensor* image_size_tensor = ctx->Tensor4ArgNameAndIndex("image_size", 0);
    const user_op::Tensor* flip_code_tensor = ctx->Tensor4ArgNameAndIndex("flip_code", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);

    int num_images = bbox_tensor->shape().elem_cnt();
    CHECK_GT(num_images, 0);
    CHECK_EQ(out_tensor->shape().elem_cnt(), num_images);
    CHECK_EQ(image_size_tensor->shape().At(0), num_images);
    CHECK_EQ(flip_code_tensor->shape().elem_cnt(), num_images);

    MultiThreadLoop(num_images, [&](size_t i) {
      const TensorBuffer& bbox_buffer = bbox_tensor->dptr<TensorBuffer>()[i];
      CHECK_EQ(bbox_buffer.shape().NumAxes(), 2);
      CHECK_EQ(bbox_buffer.shape().At(1), 4);
      TensorBuffer* out_bbox_buffer = out_tensor->mut_dptr<TensorBuffer>() + i;
      out_bbox_buffer->Resize(bbox_buffer.shape(), bbox_buffer.data_type());
      memcpy(out_bbox_buffer->mut_data(), bbox_buffer.data(), out_bbox_buffer->nbytes());
      int32_t image_height = image_size_tensor->dptr<int32_t>()[i * 2 + 0];
      int32_t image_width = image_size_tensor->dptr<int32_t>()[i * 2 + 1];
      int8_t flip_code = flip_code_tensor->dptr<int8_t>()[i];
      SwitchFlipBoxes(SwitchCase(out_bbox_buffer->data_type()), out_bbox_buffer, image_height,
                      image_width, flip_code);
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("object_bbox_flip")
    .SetCreateFn<ObjectBboxFlipKernel>()
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* bbox_tensor = ctx.TensorDesc4ArgNameAndIndex("bbox", 0);
      const user_op::TensorDesc* out_tensor = ctx.TensorDesc4ArgNameAndIndex("out", 0);
      return ctx.device_type() == DeviceType::kCPU
             && bbox_tensor->data_type() == DataType::kTensorBuffer
             && out_tensor->data_type() == DataType::kTensorBuffer;
    });

}  // namespace oneflow
