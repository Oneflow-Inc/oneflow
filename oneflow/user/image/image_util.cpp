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
#include "oneflow/user/image/image_util.h"
#include <opencv2/opencv.hpp>

namespace oneflow {

bool ImageUtil::IsColor(const std::string& color_space) {
  if (color_space == "RGB" || color_space == "BGR") {
    return true;
  } else if (color_space == "GRAY") {
    return false;
  } else {
    UNIMPLEMENTED();
    return false;
  }
}

void ImageUtil::ConvertColor(const std::string& input_color, const cv::Mat& input_img,
                             const std::string& output_color, cv::Mat& output_img) {
  if (input_color == "BGR" && output_color == "RGB") {
    cv::cvtColor(input_img, output_img, cv::COLOR_BGR2RGB);
  } else {
    UNIMPLEMENTED();
  }
}

cv::Mat GenCvMat4ImageBuffer(const TensorBuffer& image_buffer) {
  CHECK_EQ(image_buffer.shape_view().NumAxes(), 3);
  int h = image_buffer.shape_view().At(0);
  int w = image_buffer.shape_view().At(1);
  int channels = image_buffer.shape_view().At(2);
  DataType data_type = image_buffer.data_type();
  if (channels == 1 && data_type == DataType::kUInt8) {
    return CreateMatWithPtr(h, w, CV_8UC1, image_buffer.data<uint8_t>());
  } else if (channels == 1 && data_type == DataType::kFloat) {
    return CreateMatWithPtr(h, w, CV_32FC1, image_buffer.data<float>());
  } else if (channels == 3 && data_type == DataType::kUInt8) {
    return CreateMatWithPtr(h, w, CV_8UC3, image_buffer.data<uint8_t>());
  } else if (channels == 3 && data_type == DataType::kFloat) {
    return CreateMatWithPtr(h, w, CV_32FC3, image_buffer.data<float>());
  } else {
    UNIMPLEMENTED();
  }
  return cv::Mat();
}

cv::Mat GenCvMat4ImageTensor(const user_op::Tensor* image_tensor, int image_offset) {
  int has_batch_dim = 0;
  if (image_tensor->shape_view().NumAxes() == 3) {
    has_batch_dim = 0;
    image_offset = 0;
  } else if (image_tensor->shape_view().NumAxes() == 4) {
    has_batch_dim = 1;
    CHECK_GE(image_offset, 0);
    CHECK_LT(image_offset, image_tensor->shape_view().At(0));
  } else {
    UNIMPLEMENTED();
  }
  int h = image_tensor->shape_view().At(0 + has_batch_dim);
  int w = image_tensor->shape_view().At(1 + has_batch_dim);
  int c = image_tensor->shape_view().At(2 + has_batch_dim);
  int elem_offset = image_offset * h * w * c;
  DataType data_type = image_tensor->data_type();
  if (c == 1 && data_type == DataType::kUInt8) {
    return CreateMatWithPtr(h, w, CV_8UC1, image_tensor->dptr<uint8_t>() + elem_offset);
  } else if (c == 1 && data_type == DataType::kFloat) {
    return CreateMatWithPtr(h, w, CV_32FC1, image_tensor->dptr<float>() + elem_offset);
  } else if (c == 3 && data_type == DataType::kUInt8) {
    return CreateMatWithPtr(h, w, CV_8UC3, image_tensor->dptr<uint8_t>() + elem_offset);
  } else if (c == 3 && data_type == DataType::kFloat) {
    return CreateMatWithPtr(h, w, CV_32FC3, image_tensor->dptr<float>() + elem_offset);
  } else {
    UNIMPLEMENTED();
  }
  return cv::Mat();
}

void CvMatConvertToDataType(const cv::Mat& src, cv::Mat* dst, DataType dtype) {
  if (dtype == DataType::kUInt8) {
    src.convertTo(*dst, CV_8U);
  } else if (dtype == DataType::kFloat) {
    src.convertTo(*dst, CV_32F);
  } else {
    UNIMPLEMENTED();
  }
}

int GetCvInterpolationFlag(const std::string& interp_type, int org_w, int org_h, int res_w,
                           int res_h) {
  if (interp_type == "bilinear") {
    return cv::INTER_LINEAR;
  } else if (interp_type == "nearest_neighbor" || interp_type == "nn") {
    return cv::INTER_NEAREST;
  } else if (interp_type == "bicubic") {
    return cv::INTER_CUBIC;
  } else if (interp_type == "area") {
    return cv::INTER_AREA;
  } else if (interp_type == "auto") {
    if (res_w * res_h >= org_w * org_h) {
      return cv::INTER_LINEAR;
    } else {
      return cv::INTER_AREA;
    }
  } else {
    UNIMPLEMENTED();
  }
}

bool CheckInterpolationValid(const std::string& interp_type, std::ostringstream& err) {
  if (interp_type != "bilinear" && interp_type != "nearest_neighbor" && interp_type != "nn"
      && interp_type != "bicubic" && interp_type != "area" && interp_type != "auto") {
    err << ", interpolation_type: " << interp_type
        << " (interpolation_type must be one of bilinear, nearest_neighbor(nn), bicubic, area and "
           "auto)";
    return false;
  }
  return true;
}

}  // namespace oneflow
