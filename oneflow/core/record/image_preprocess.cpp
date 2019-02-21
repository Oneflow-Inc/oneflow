#include "oneflow/core/record/image_preprocess.h"

namespace oneflow {

namespace {

float GetRandomFloatValue(float min, float max, std::function<int32_t(void)> NextRandomInt) {
  float ratio = static_cast<float>(NextRandomInt()) / static_cast<float>(GetMaxVal<int32_t>());
  return (max - min) * ratio + min;
}

}  // namespace

void ImagePreprocessImpl<PreprocessCase::kResize>::DoPreprocess(
    cv::Mat* image, const ImagePreprocess& preprocess_conf,
    std::function<int32_t(void)> NextRandomInt) const {
  CHECK(preprocess_conf.has_resize());
  const ImageResize& conf = preprocess_conf.resize();
  cv::Mat dst;
  cv::resize(*image, dst, cv::Size(conf.width(), conf.height()), 0, 0, cv::INTER_LINEAR);
  *image = dst;
}

void ImagePreprocessImpl<PreprocessCase::kCrop>::DoPreprocess(
    cv::Mat* image, const ImagePreprocess& preprocess_conf,
    std::function<int32_t(void)> NextRandomInt) const {
  CHECK(preprocess_conf.has_crop());
  const ImageCrop& crop = preprocess_conf.crop();
  int32_t x = crop.x();
  int32_t y = crop.y();
  int32_t width = crop.width();
  int32_t height = crop.height();
  CHECK_LE(width, image->cols);
  CHECK_LE(height, image->rows);
  if (crop.random_xy()) {
    int32_t x_max = (image->cols - width);
    int32_t y_max = (image->rows - height);
    x = x_max > 0 ? (NextRandomInt() % x_max) : x_max;
    y = y_max > 0 ? (NextRandomInt() % y_max) : y_max;
  } else {
    CHECK_LE(x, image->cols - width);
    CHECK_LE(y, image->rows - height);
  }
  *image = (*image)(cv::Rect(x, y, width, height));
}

void ImagePreprocessImpl<PreprocessCase::kCropWithRandomSize>::DoPreprocess(
    cv::Mat* image, const ImagePreprocess& preprocess_conf,
    std::function<int32_t(void)> NextRandomInt) const {
  CHECK(preprocess_conf.has_crop_with_random_size());
  const ImageCropWithRandomSize& conf = preprocess_conf.crop_with_random_size();
  int32_t max_attempts = conf.max_attempts();
  float area_min = conf.area_range().min();
  float area_max = conf.area_range().max();
  float ratio_min = conf.aspect_ratio_range().min();
  float ratio_max = conf.aspect_ratio_range().max();
  CHECK_LE(area_min, area_max);
  CHECK_GT(area_min, 0.0);
  CHECK_LE(area_max, 1.0);
  CHECK_LE(ratio_min, ratio_max);
  CHECK_GT(ratio_min, 0.0);
  while (max_attempts--) {
    float area_size = GetRandomFloatValue(area_min, area_max, NextRandomInt) * image->total();
    float aspect_ratio = GetRandomFloatValue(ratio_min, ratio_max, NextRandomInt);
    float height_float = sqrt(area_size / aspect_ratio);
    int32_t height = static_cast<int32_t>(height_float);
    int32_t width = static_cast<int32_t>(height_float * aspect_ratio);
    if (width <= image->cols && height <= image->rows) {
      int32_t x_max = (image->cols - width);
      int32_t y_max = (image->rows - height);
      int32_t x = x_max > 0 ? (NextRandomInt() % x_max) : x_max;
      int32_t y = y_max > 0 ? (NextRandomInt() % y_max) : y_max;
      *image = (*image)(cv::Rect(x, y, width, height));
      return;
    }
  }
}

void ImagePreprocessImpl<PreprocessCase::kMirror>::DoPreprocess(
    cv::Mat* image, const ImagePreprocess& preprocess_conf,
    std::function<int32_t(void)> NextRandomInt) const {
  CHECK(preprocess_conf.has_mirror());
  if (NextRandomInt() % 2 == 0) {
    // %50 mirror probability
    return;
  }
  cv::Mat dst;
  cv::flip(*image, dst, 1);
  *image = dst;
}

ImagePreprocessIf* GetImagePreprocess(PreprocessCase preprocess_case) {
  static const HashMap<int, ImagePreprocessIf*> obj = {

#define MAKE_ENTRY(preprocess_case) {preprocess_case, new ImagePreprocessImpl<preprocess_case>},
      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, PREPROCESS_CASE_SEQ)};
  return obj.at(preprocess_case);
}

}  // namespace oneflow
