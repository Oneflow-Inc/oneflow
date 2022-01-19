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
#ifndef ONEFLOW_USER_IMAGE_IMAGE_UTIL_H_
#define ONEFLOW_USER_IMAGE_IMAGE_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/framework/user_op_tensor.h"
#include <opencv2/opencv.hpp>

namespace oneflow {

struct ImageUtil {
  static bool IsColor(const std::string& color_space);

  static void ConvertColor(const std::string& input_color, const cv::Mat& input_img,
                           const std::string& output_color, cv::Mat& output_img);
};

template<typename T>
inline cv::Mat CreateMatWithPtr(int H, int W, int type, const T* ptr,
                                size_t step = cv::Mat::AUTO_STEP) {
  return cv::Mat(H, W, type, const_cast<T*>(ptr), step);
}

cv::Mat GenCvMat4ImageBuffer(const TensorBuffer& image_buffer);

cv::Mat GenCvMat4ImageTensor(const user_op::Tensor* image_tensor, int image_offset);

void CvMatConvertToDataType(const cv::Mat& src, cv::Mat* dst, DataType dtype);

int GetCvInterpolationFlag(const std::string& inter_type, int org_w, int org_h, int res_w,
                           int res_h);
bool CheckInterpolationValid(const std::string& interp_type, std::ostringstream& ss);

}  // namespace oneflow

#endif  // ONEFLOW_USER_IMAGE_IMAGE_UTIL_H_
