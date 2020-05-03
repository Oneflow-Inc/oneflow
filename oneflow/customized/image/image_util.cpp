#include "oneflow/customized/image/image_util.h"
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

}  // namespace oneflow
