#include "oneflow/core/record/image_preprocess.h"

namespace oneflow {

void ImagePreprocessImpl<PreprocessCase::kResize>::DoPreprocess(
    cv::Mat* image, const ImagePreprocess& preprocess_conf,
    std::mt19937* random) const {
  CHECK(preprocess_conf.has_resize());
  const ImageSize& size = preprocess_conf.resize().size();
  cv::Mat dst;
  cv::resize(*image, dst, cv::Size(size.width(), size.height()), 0, 0,
             cv::INTER_LINEAR);
  *image = dst;
}

void ImagePreprocessImpl<PreprocessCase::kCrop>::DoPreprocess(
    cv::Mat* image, const ImagePreprocess& preprocess_conf,
    std::mt19937* random) const {
  CHECK(preprocess_conf.has_crop());
  const ImageCrop& crop = preprocess_conf.crop();
  int32_t x = crop.x();
  int32_t y = crop.y();
  int32_t width = crop.size().width();
  int32_t height = crop.size().height();
  CHECK_LE(width, image->cols);
  CHECK_LE(height, image->rows);
  if (crop.random_xy()) {
    x = (*random)() % (image->cols - width);
    y = (*random)() % (image->rows - height);
  } else {
    CHECK_LE(x, image->cols - width);
    CHECK_LE(y, image->rows - height);
  }
  *image = (*image)(cv::Rect(x, y, width, height));
}

void ImagePreprocessImpl<PreprocessCase::kMirror>::DoPreprocess(
    cv::Mat* image, const ImagePreprocess& preprocess_conf,
    std::mt19937* random) const {
  CHECK(preprocess_conf.has_mirror());
  if ((*random)() % 2 == 0) {
    // %50 mirror probability
    return;
  }
  cv::Mat dst;
  cv::flip(*image, dst, 1);
  *image = dst;
}

ImagePreprocessIf* GetImagePreprocess(PreprocessCase preprocess_case) {
  static const HashMap<int, ImagePreprocessIf*> obj = {

#define MAKE_ENTRY(preprocess_case) \
  {preprocess_case, new ImagePreprocessImpl<preprocess_case>},
      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, PREPROCESS_CASE_SEQ)};
  return obj.at(preprocess_case);
}

}  // namespace oneflow
