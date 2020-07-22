#include "oneflow/core/framework/framework.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/customized/image/image_util.h"
#include <opencv2/opencv.hpp>
#include <cfenv>

namespace oneflow {

namespace {

template<typename T>
std::pair<T, T> GetTargetResizedSize4ImageBuffer(const TensorBuffer& image_buffer,
                                                 const T target_size, const T max_size) {
  CHECK_EQ(image_buffer.shape().NumAxes(), 3);
  const T origin_height = image_buffer.shape().At(0);
  const T origin_width = image_buffer.shape().At(1);

  // set round to banker's rounding
  int origin_round_way = std::fegetround();
  CHECK_EQ(std::fesetround(FE_TONEAREST), 0);

  double origin_min_size = std::min<double>(origin_height, origin_width);
  double origin_max_size = std::max<double>(origin_height, origin_width);
  double resized_min_size = static_cast<double>(target_size);
  double resized_max_size = std::nearbyint((origin_max_size / origin_min_size) * resized_min_size);
  if (max_size > 0 && resized_max_size > max_size) {
    resized_max_size = static_cast<double>(max_size);
    resized_min_size = std::nearbyint(resized_max_size * origin_min_size / origin_max_size);
  }

  std::pair<T, T> width_and_height;
  if (origin_width < origin_height) {
    width_and_height.first = resized_min_size;
    width_and_height.second = resized_max_size;
  } else {
    width_and_height.first = resized_max_size;
    width_and_height.second = resized_min_size;
  }
  std::fesetround(origin_round_way);
  return width_and_height;
}

void ImageTargetResize(const TensorBuffer& image_buffer, TensorBuffer* resized_image_buffer,
                       const int32_t target_size, const int32_t max_size,
                       const std::string& interpolation) {
  CHECK_EQ(image_buffer.shape().NumAxes(), 3);
  CHECK_GT(target_size, 0);
  CHECK(max_size >= target_size || max_size == 0);

  const cv::Mat image_mat = GenCvMat4ImageBuffer(image_buffer);
  int64_t res_w = 0;
  int64_t res_h = 0;
  int64_t channels = image_mat.channels();
  std::tie(res_w, res_h) =
      GetTargetResizedSize4ImageBuffer<int64_t>(image_buffer, target_size, max_size);
  resized_image_buffer->Resize(Shape({res_h, res_w, channels}), image_buffer.data_type());
  cv::Mat res_image_mat = GenCvMat4ImageBuffer(*resized_image_buffer);
  int interpolaion_flag =
      GetCvInterpolationFlag(interpolation, image_mat.cols, image_mat.rows, res_w, res_h);
  cv::resize(image_mat, res_image_mat, cv::Size(res_w, res_h), 0, 0, interpolaion_flag);

  CHECK_EQ(res_image_mat.ptr(), resized_image_buffer->data());
  CHECK_LE(std::max(res_image_mat.rows, res_image_mat.cols), max_size);
  CHECK(std::max(res_image_mat.rows, res_image_mat.cols) == max_size
        || std::min(res_image_mat.rows, res_image_mat.cols) == target_size);
}

}  // namespace

class ImageTargetResizeKernel final : public user_op::OpKernel {
 public:
  ImageTargetResizeKernel() = default;
  ~ImageTargetResizeKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* size_tensor = ctx->Tensor4ArgNameAndIndex("size", 0);
    user_op::Tensor* scale_tensor = ctx->Tensor4ArgNameAndIndex("scale", 0);
    CHECK_GT(in_tensor->shape().elem_cnt(), 0);
    CHECK_EQ(in_tensor->shape().elem_cnt(), out_tensor->shape().elem_cnt());
    CHECK_EQ(in_tensor->shape().elem_cnt(), size_tensor->shape().At(0));
    CHECK_EQ(in_tensor->shape().elem_cnt(), scale_tensor->shape().At(0));

    const TensorBuffer* in_img_buf = in_tensor->dptr<TensorBuffer>();
    TensorBuffer* out_img_buf = out_tensor->mut_dptr<TensorBuffer>();
    int32_t* size_ptr = size_tensor ? size_tensor->mut_dptr<int32_t>() : nullptr;
    float* scale_ptr = scale_tensor ? scale_tensor->mut_dptr<float>() : nullptr;
    const int32_t target_size = ctx->Attr<int32_t>("target_size");
    const int32_t max_size = ctx->Attr<int32_t>("max_size");

    MultiThreadLoop(in_tensor->shape().elem_cnt(), [&](size_t i) {
      ImageTargetResize(in_img_buf[i], out_img_buf + i, target_size, max_size,
                        ctx->Attr<std::string>("interpolation"));
      const int64_t org_h = in_img_buf[i].shape().At(0);
      const int64_t org_w = in_img_buf[i].shape().At(1);
      const int64_t res_h = out_img_buf[i].shape().At(0);
      const int64_t res_w = out_img_buf[i].shape().At(1);
      if (size_ptr != nullptr) {
        size_ptr[i * 2 + 0] = static_cast<int32_t>(res_w);
        size_ptr[i * 2 + 1] = static_cast<int32_t>(res_h);
      }
      if (scale_ptr != nullptr) {
        scale_ptr[i * 2 + 0] = static_cast<float>(res_w) / static_cast<float>(org_w);
        scale_ptr[i * 2 + 1] = static_cast<float>(res_h) / static_cast<float>(org_h);
      }
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("image_target_resize")
    .SetCreateFn<ImageTargetResizeKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)
                     & (user_op::HobDataType("in", 0) == DataType::kTensorBuffer)
                     & (user_op::HobDataType("out", 0) == DataType::kTensorBuffer)
                     & (user_op::HobDataType("size", 0) == DataType::kInt32)
                     & (user_op::HobDataType("scale", 0) == DataType::kFloat));

}  // namespace oneflow
