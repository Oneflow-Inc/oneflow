#include "oneflow/core/record/image_preprocess.h"

namespace oneflow {

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
    x = NextRandomInt() % (image->cols - width);
    y = NextRandomInt() % (image->rows - height);
  } else {
    CHECK_LE(x, image->cols - width);
    CHECK_LE(y, image->rows - height);
  }
  *image = (*image)(cv::Rect(x, y, width, height));
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

void ImagePreprocessImpl<PreprocessCase::kTargetResize>::DoPreprocess(
    cv::Mat* image, const ImagePreprocess& preprocess_conf,
    std::function<int32_t(void)> NextRandomInt) const {
  CHECK(preprocess_conf.has_target_resize());
  const int32_t target_size = preprocess_conf.target_resize().target_size();
  const int32_t max_size = preprocess_conf.target_resize().max_size();
  CHECK_GT(target_size, 0);
  CHECK_GE(max_size, target_size);
  const int32_t im_size_min = std::min(image->rows, image->cols);
  const int32_t im_size_max = std::max(image->rows, image->cols);
  float im_scale = static_cast<float>(target_size) / static_cast<float>(im_size_min);
  if (std::round(im_scale * im_size_max) > max_size) {
    im_scale = static_cast<float>(max_size) / static_cast<float>(im_size_max);
  }
  cv::resize(*image, *image, cv::Size(), im_scale, im_scale, cv::INTER_LINEAR);
  CHECK_LE(std::max(image->rows, image->cols), max_size);
  CHECK(std::max(image->rows, image->cols) == max_size
        || std::min(image->rows, image->cols) == target_size);
}

ImagePreprocessIf* GetImagePreprocess(PreprocessCase preprocess_case) {
  static const HashMap<int, ImagePreprocessIf*> obj = {

#define MAKE_ENTRY(preprocess_case) {preprocess_case, new ImagePreprocessImpl<preprocess_case>},
      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, PREPROCESS_CASE_SEQ)};
  return obj.at(preprocess_case);
}

}  // namespace oneflow
