/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/user/image/image_util.h"
#include <opencv2/opencv.hpp>
#include <cfenv>

namespace oneflow {

namespace {

template<typename T>
std::pair<T, T> GetTargetResizedSize4ImageBuffer(const TensorBuffer& image_buffer,
                                                 const bool resize_longer, const T target_size,
                                                 const T min_size, const T max_size) {
  CHECK_GT(target_size, 0);
  if (min_size > 0) { CHECK_GE(target_size, min_size); }
  if (max_size > 0) { CHECK_LE(target_size, max_size); }
  CHECK_EQ(image_buffer.shape_view().NumAxes(), 3);
  const T origin_height = image_buffer.shape_view().At(0);
  const T origin_width = image_buffer.shape_view().At(1);

  // set round to banker's rounding
  int origin_round_way = std::fegetround();
  CHECK_EQ(std::fesetround(FE_TONEAREST), 0);

  double org_min_size = std::min<double>(origin_height, origin_width);
  double org_max_size = std::max<double>(origin_height, origin_width);
  double aspect_ratio = org_min_size / org_max_size;
  double res_min_size = 0.0;
  double res_max_size = 0.0;
  if (resize_longer) {
    res_max_size = static_cast<double>(target_size);
    res_min_size = std::nearbyint(res_max_size * aspect_ratio);
    if (min_size > 0 && res_min_size < min_size) {
      res_min_size = static_cast<double>(min_size);
      res_max_size = std::nearbyint(res_min_size / aspect_ratio);
    }
  } else {
    res_min_size = static_cast<double>(target_size);
    res_max_size = std::nearbyint(res_min_size / aspect_ratio);
    if (max_size > 0 && res_max_size > max_size) {
      res_max_size = static_cast<double>(max_size);
      res_min_size = std::nearbyint(res_max_size * aspect_ratio);
    }
  }
  std::fesetround(origin_round_way);

  std::pair<T, T> width_and_height;
  if (origin_width < origin_height) {
    width_and_height.first = static_cast<T>(res_min_size);
    width_and_height.second = static_cast<T>(res_max_size);
  } else {
    width_and_height.first = static_cast<T>(res_max_size);
    width_and_height.second = static_cast<T>(res_min_size);
  }
  return width_and_height;
}

bool CheckMatSizeMatch(const cv::Mat& mat, const bool resize_longer, const int32_t target_size,
                       const int32_t min_size, const int32_t max_size) {
  bool is_size_match = true;
  int mat_min_size = std::min(mat.rows, mat.cols);
  int mat_max_size = std::max(mat.rows, mat.cols);
  if (resize_longer) {
    if (min_size > 0) {
      is_size_match = (mat_max_size >= target_size) && (mat_min_size >= min_size)
                      && (mat_min_size == min_size || mat_max_size == target_size);
    } else {
      is_size_match = (mat_max_size == target_size);
    }
  } else {
    if (max_size > 0) {
      is_size_match = (mat_min_size <= target_size) && (mat_max_size <= max_size)
                      && (mat_min_size == target_size || mat_max_size == max_size);
    } else {
      is_size_match = (mat_min_size == target_size);
    }
  }
  return is_size_match;
}

void ImageTargetResize(const TensorBuffer& image_buffer, TensorBuffer* resized_image_buffer,
                       const bool resize_longer, const int32_t target_size, const int32_t min_size,
                       const int32_t max_size, const std::string& interp_type) {
  const cv::Mat image_mat = GenCvMat4ImageBuffer(image_buffer);
  int64_t res_w = 0;
  int64_t res_h = 0;
  int64_t channels = image_mat.channels();
  std::tie(res_w, res_h) = GetTargetResizedSize4ImageBuffer<int64_t>(
      image_buffer, resize_longer, target_size, min_size, max_size);
  resized_image_buffer->Resize(Shape({res_h, res_w, channels}), image_buffer.data_type());
  cv::Mat res_image_mat = GenCvMat4ImageBuffer(*resized_image_buffer);
  int interp_flag =
      GetCvInterpolationFlag(interp_type, image_mat.cols, image_mat.rows, res_w, res_h);
  cv::resize(image_mat, res_image_mat, cv::Size(res_w, res_h), 0, 0, interp_flag);

  CHECK_EQ(res_image_mat.ptr<void>(), resized_image_buffer->data());
  CHECK(CheckMatSizeMatch(res_image_mat, resize_longer, target_size, min_size, max_size));
}

class ImageResizeToFixedSizeKernel final : public user_op::OpKernel {
 public:
  ImageResizeToFixedSizeKernel() = default;
  ~ImageResizeToFixedSizeKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    CHECK_NOTNULL(in_tensor);
    const int64_t batch_size = in_tensor->shape_view().elem_cnt();
    CHECK_GT(batch_size, 0);

    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(out_tensor->shape_view().NumAxes(), 4);
    CHECK_EQ(out_tensor->shape_view().At(0), batch_size);
    int64_t res_h = out_tensor->shape_view().At(1);
    int64_t res_w = out_tensor->shape_view().At(2);
    int64_t channels = out_tensor->shape_view().At(3);
    int64_t elem_cnt_per_img = res_h * res_w * channels;

    user_op::Tensor* scale_tensor = ctx->Tensor4ArgNameAndIndex("scale", 0);
    CHECK_EQ(scale_tensor->shape_view().NumAxes(), 2);
    CHECK_EQ(scale_tensor->shape_view().At(0), batch_size);
    CHECK_EQ(scale_tensor->shape_view().At(1), 2);

    MultiThreadLoop(batch_size, [&](size_t i) {
      const TensorBuffer& in_buffer = in_tensor->dptr<TensorBuffer>()[i];
      CHECK_EQ(in_buffer.shape_view().NumAxes(), 3);
      const int64_t origin_height = in_buffer.shape_view().At(0);
      const int64_t origin_width = in_buffer.shape_view().At(1);
      CHECK_EQ(in_buffer.shape_view().At(2), channels);
      DataType dtype = ctx->Attr<DataType>("data_type");
      int interp_flag = GetCvInterpolationFlag(ctx->Attr<std::string>("interpolation_type"),
                                               origin_width, origin_height, res_w, res_h);

      const cv::Mat in_img_mat = GenCvMat4ImageBuffer(in_buffer);
      cv::Mat out_img_mat = GenCvMat4ImageTensor(out_tensor, i);
      if (in_buffer.data_type() == dtype) {
        cv::resize(in_img_mat, out_img_mat, cv::Size(res_w, res_h), 0, 0, interp_flag);
      } else {
        cv::Mat res_img_mat;
        cv::resize(in_img_mat, res_img_mat, cv::Size(res_w, res_h), 0, 0, interp_flag);
        CvMatConvertToDataType(res_img_mat, &out_img_mat, dtype);
      }

      char* cur_out_dptr =
          out_tensor->mut_dptr<char>() + i * elem_cnt_per_img * GetSizeOfDataType(dtype);
      CHECK(out_img_mat.isContinuous());
      CHECK_EQ(out_img_mat.ptr<void>(), static_cast<void*>(cur_out_dptr));
      CHECK_EQ(out_img_mat.cols, res_w);
      CHECK_EQ(out_img_mat.rows, res_h);
      CHECK_EQ(out_img_mat.channels(), channels);

      if (scale_tensor) {
        float* scale_dptr = scale_tensor->mut_dptr<float>() + i * 2;
        scale_dptr[0] = static_cast<float>(res_w) / static_cast<float>(origin_width);
        scale_dptr[1] = static_cast<float>(res_h) / static_cast<float>(origin_height);
      }
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class ImageResizeKeepAspectRatioKernel final : public user_op::OpKernel {
 public:
  ImageResizeKeepAspectRatioKernel() = default;
  ~ImageResizeKeepAspectRatioKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* size_tensor = ctx->Tensor4ArgNameAndIndex("size", 0);
    user_op::Tensor* scale_tensor = ctx->Tensor4ArgNameAndIndex("scale", 0);
    CHECK_NOTNULL(out_tensor);
    CHECK_NOTNULL(size_tensor);
    CHECK_NOTNULL(scale_tensor);
    const TensorBuffer* in_img_buf = in_tensor->dptr<TensorBuffer>();
    TensorBuffer* out_img_buf = out_tensor->mut_dptr<TensorBuffer>();
    TensorBuffer* scale_buf = scale_tensor->mut_dptr<TensorBuffer>();
    TensorBuffer* size_buf = size_tensor->mut_dptr<TensorBuffer>();

    const int64_t num_images = in_tensor->shape_view().elem_cnt();
    const bool resize_longer = ctx->Attr<bool>("resize_longer");
    const int32_t target_size = ctx->Attr<int32_t>("target_size");
    const int32_t min_size = ctx->Attr<int32_t>("min_size");
    const int32_t max_size = ctx->Attr<int32_t>("max_size");
    const std::string& interp_type = ctx->Attr<std::string>("interpolation_type");

    MultiThreadLoop(num_images, [&](size_t i) {
      ImageTargetResize(in_img_buf[i], out_img_buf + i, resize_longer, target_size, min_size,
                        max_size, interp_type);
      const int64_t org_h = in_img_buf[i].shape_view().At(0);
      const int64_t org_w = in_img_buf[i].shape_view().At(1);
      const int64_t res_h = out_img_buf[i].shape_view().At(0);
      const int64_t res_w = out_img_buf[i].shape_view().At(1);

      scale_buf[i].Resize(Shape({2}), DataType::kFloat);
      scale_buf[i].mut_data<float>()[0] = static_cast<float>(res_w) / static_cast<float>(org_w);
      scale_buf[i].mut_data<float>()[1] = static_cast<float>(res_h) / static_cast<float>(org_h);

      size_buf[i].Resize(Shape({2}), DataType::kInt32);
      size_buf[i].mut_data<int32_t>()[0] = static_cast<int32_t>(res_w);
      size_buf[i].mut_data<int32_t>()[1] = static_cast<int32_t>(res_h);
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_IMAGE_RESIZE_KERNEL(dtype)                                          \
  REGISTER_USER_KERNEL("image_resize_to_fixed")                                      \
      .SetCreateFn<ImageResizeToFixedSizeKernel>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                \
                       && (user_op::HobDataType("in", 0) == DataType::kTensorBuffer) \
                       && (user_op::HobAttr<DataType>("data_type") == GetDataType<dtype>::value));

REGISTER_IMAGE_RESIZE_KERNEL(float)
REGISTER_IMAGE_RESIZE_KERNEL(uint8_t)

REGISTER_USER_KERNEL("image_resize_keep_aspect_ratio")
    .SetCreateFn<ImageResizeKeepAspectRatioKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)
                     && (user_op::HobDataType("in", 0) == DataType::kTensorBuffer)
                     && (user_op::HobDataType("out", 0) == DataType::kTensorBuffer)
                     && (user_op::HobDataType("size", 0) == DataType::kTensorBuffer)
                     && (user_op::HobDataType("scale", 0) == DataType::kTensorBuffer));

}  // namespace oneflow
