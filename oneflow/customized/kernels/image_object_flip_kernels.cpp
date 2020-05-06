#include "oneflow/core/framework/framework.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/customized/image/image_util.h"
#include <opencv2/opencv.hpp>

namespace oneflow {

namespace {

enum class FlipCode : int8_t {
  kNonFlip = 0x00,
  kHorizontalFlip = 0x01,
  kVerticalFlip = 0x10,
  kBothDirectionFlip = 0x11,
};

bool operator&(FlipCode lhs, FlipCode rhs) {
  return static_cast<bool>(static_cast<std::underlying_type<FlipCode>::type>(lhs)
                           & static_cast<std::underlying_type<FlipCode>::type>(rhs));
}

int CvFlipCode(FlipCode flip_code) {
  if (flip_code == FlipCode::kHorizontalFlip) {
    return 1;
  } else if (flip_code == FlipCode::kVerticalFlip) {
    return 0;
  } else if (flip_code == FlipCode::kBothDirectionFlip) {
    return -1;
  } else {
    UNIMPLEMENTED();
  }
}

void FlipImage(TensorBuffer* image_buffer, FlipCode flip_code) {
  cv::Mat image_mat = GenCvMat4ImageBuffer(*image_buffer);
  cv::flip(image_mat, image_mat, CvFlipCode(flip_code));
}

template<typename T>
void FlipBoxes(TensorBuffer* boxes_buffer, int32_t image_height, int32_t image_width,
               FlipCode flip_code) {
  int num_boxes = boxes_buffer->shape().At(0);
  FOR_RANGE(int, i, 0, num_boxes) {
    T* cur_box_ptr = boxes_buffer->mut_data<T>() + i * 4;
    if (flip_code & FlipCode::kHorizontalFlip) {
      T xmin = cur_box_ptr[0];
      T xmax = cur_box_ptr[2];
      cur_box_ptr[0] = image_width - xmax - static_cast<T>(1);
      cur_box_ptr[2] = image_width - xmin - static_cast<T>(1);
    }
    if (flip_code & FlipCode::kVerticalFlip) {
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

template<typename T>
void FlipPolygons(TensorBuffer* polygons_buffer, int32_t image_height, int32_t image_width,
                  FlipCode flip_code) {
  int num_points = polygons_buffer->shape().At(0);
  FOR_RANGE(int, i, 0, num_points) {
    T* cur_poly_ptr = polygons_buffer->mut_data<T>() + i * 2;
    if (flip_code & FlipCode::kHorizontalFlip) { cur_poly_ptr[0] = image_width - cur_poly_ptr[0]; }
    if (flip_code & FlipCode::kVerticalFlip) { cur_poly_ptr[1] = image_height - cur_poly_ptr[1]; }
  }
}

#define MAKE_FLIP_POLYGONS_SWITCH_ENTRY(func_name, T) func_name<T>
DEFINE_STATIC_SWITCH_FUNC(void, FlipPolygons, MAKE_FLIP_POLYGONS_SWITCH_ENTRY,
                          MAKE_DATA_TYPE_CTRV_SEQ(FLOATING_DATA_TYPE_SEQ));

#undef MAKE_FLIP_POLYGONS_SWITCH_ENTRY

std::function<bool(const user_op::KernelRegContext&)> MakeKernelMatchPredFn(
    const std::string& input_arg_name) {
  return [&](const user_op::KernelRegContext& ctx) -> bool {
    const user_op::TensorDesc* in_tensor = ctx.TensorDesc4ArgNameAndIndex(input_arg_name, 0);
    const user_op::TensorDesc* out_tensor = ctx.TensorDesc4ArgNameAndIndex("out", 0);
    return ctx.device_type() == DeviceType::kCPU
           && in_tensor->data_type() == DataType::kTensorBuffer
           && out_tensor->data_type() == DataType::kTensorBuffer;
  };
}

}  // namespace

class ImageFlipKernel final : public user_op::OpKernel {
 public:
  ImageFlipKernel() = default;
  ~ImageFlipKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* flip_code_tensor = ctx->Tensor4ArgNameAndIndex("flip_code", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    int num_images = in_tensor->shape().elem_cnt();
    CHECK_EQ(out_tensor->shape().elem_cnt(), num_images);

    MultiThreadLoop(num_images, [&](size_t i) {
      const TensorBuffer& in_buffer = in_tensor->dptr<TensorBuffer>()[i];
      CHECK_EQ(in_buffer.shape().NumAxes(), 3);
      TensorBuffer* out_buffer = out_tensor->mut_dptr<TensorBuffer>() + i;
      out_buffer->Resize(in_buffer.shape(), in_buffer.data_type());
      memcpy(out_buffer->mut_data(), in_buffer.data(), out_buffer->nbytes());
      FlipCode flip_code = static_cast<FlipCode>(flip_code_tensor->dptr<int8_t>()[i]);
      if (flip_code != FlipCode::kNonFlip) { FlipImage(out_buffer, flip_code); }
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

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
      FlipCode flip_code = static_cast<FlipCode>(flip_code_tensor->dptr<int8_t>()[i]);
      SwitchFlipBoxes(SwitchCase(out_bbox_buffer->data_type()), out_bbox_buffer, image_height,
                      image_width, flip_code);
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class ObjectSegmentationPolygonFlipKernel final : public user_op::OpKernel {
 public:
  ObjectSegmentationPolygonFlipKernel() = default;
  ~ObjectSegmentationPolygonFlipKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* polygon_tensor = ctx->Tensor4ArgNameAndIndex("polygon", 0);
    const user_op::Tensor* image_size_tensor = ctx->Tensor4ArgNameAndIndex("image_size", 0);
    const user_op::Tensor* flip_code_tensor = ctx->Tensor4ArgNameAndIndex("flip_code", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);

    int num_images = polygon_tensor->shape().elem_cnt();
    CHECK_GT(num_images, 0);
    CHECK_EQ(out_tensor->shape().elem_cnt(), num_images);
    CHECK_EQ(image_size_tensor->shape().At(0), num_images);
    CHECK_EQ(flip_code_tensor->shape().elem_cnt(), num_images);

    MultiThreadLoop(num_images, [&](size_t i) {
      const TensorBuffer& polygons_buffer = polygon_tensor->dptr<TensorBuffer>()[i];
      CHECK_EQ(polygons_buffer.shape().NumAxes(), 2);
      CHECK_EQ(polygons_buffer.shape().At(1), 2);
      TensorBuffer* out_polygons_buffer = out_tensor->mut_dptr<TensorBuffer>() + i;
      out_polygons_buffer->Resize(polygons_buffer.shape(), polygons_buffer.data_type());
      memcpy(out_polygons_buffer->mut_data(), polygons_buffer.data(),
             out_polygons_buffer->nbytes());
      int32_t image_height = image_size_tensor->dptr<int32_t>()[i * 2 + 0];
      int32_t image_width = image_size_tensor->dptr<int32_t>()[i * 2 + 1];
      FlipCode flip_code = static_cast<FlipCode>(flip_code_tensor->dptr<int8_t>()[i]);
      SwitchFlipPolygons(SwitchCase(out_polygons_buffer->data_type()), out_polygons_buffer,
                         image_height, image_width, flip_code);
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("image_flip")
    .SetCreateFn<ImageFlipKernel>()
    .SetIsMatchedPred(MakeKernelMatchPredFn("image"));

REGISTER_USER_KERNEL("object_bbox_flip")
    .SetCreateFn<ObjectBboxFlipKernel>()
    .SetIsMatchedPred(MakeKernelMatchPredFn("bbox"));

REGISTER_USER_KERNEL("object_segmentation_polygon_flip")
    .SetCreateFn<ObjectSegmentationPolygonFlipKernel>()
    .SetIsMatchedPred(MakeKernelMatchPredFn("polygon"));

}  // namespace oneflow
