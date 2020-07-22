#ifndef ONEFLOW_CUSTOMIZED_IMAGE_IMAGE_UTIL_H_
#define ONEFLOW_CUSTOMIZED_IMAGE_IMAGE_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/framework/tensor.h"
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

int GetCvInterpolationFlag(const std::string& inter_type, int org_w, int org_h, int res_w,
                           int res_h);
bool CheckInterpolationValid(const std::string& interpolation, std::ostringstream& ss);

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_IMAGE_IMAGE_UTIL_H_
