#ifndef ONEFLOW_CUSTOMIZED_IMAGE_IMAGE_UTIL_H_
#define ONEFLOW_CUSTOMIZED_IMAGE_IMAGE_UTIL_H_

#include "oneflow/core/common/util.h"
#include <opencv2/opencv.hpp>

namespace oneflow {

struct ImageUtil {
  static bool IsColor(const std::string& color_space);

  static void ConvertColor(const std::string& input_color, const cv::Mat& input_img,
                           const std::string& output_color, cv::Mat& output_img);
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_IMAGE_IMAGE_UTIL_H_
