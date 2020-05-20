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

cv::Mat GenCvMat4ImageBuffer(const TensorBuffer& image_buffer) {
  CHECK_EQ(image_buffer.shape().NumAxes(), 3);
  int h = image_buffer.shape().At(0);
  int w = image_buffer.shape().At(1);
  int channels = image_buffer.shape().At(2);
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

}  // namespace oneflow
