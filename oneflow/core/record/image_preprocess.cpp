#include "oneflow/core/record/image_preprocess.h"

namespace oneflow {

void ImagePreprocessImpl<PreprocessCase::kResize>::DoPreprocess(
    cv::Mat* image, const ImagePreprocess& preprocess_conf) const {
  CHECK(preprocess_conf.has_resize());
  const ImageSize& size = preprocess_conf.resize().size();
  cv::Mat dst;
  cv::resize(*image, dst, cv::Size(size.width(), size.height()), 0, 0,
             cv::INTER_LINEAR);
  *image = dst;
}

void ImagePreprocessImpl<PreprocessCase::kCrop>::DoPreprocess(
    cv::Mat* image, const ImagePreprocess& preprocess_conf) const {
  CHECK(preprocess_conf.has_crop());
  const ImageCrop& crop = preprocess_conf.crop();
  *image = (*image)(
      cv::Rect(crop.x(), crop.y(), crop.size().width(), crop.size().height()));
}

ImagePreprocessIf* GetImagePreprocess(PreprocessCase preprocess_case) {
  static const HashMap<int, ImagePreprocessIf*> obj = {

#define MAKE_ENTRY(preprocess_case) \
  {preprocess_case, new ImagePreprocessImpl<preprocess_case>},
      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, PREPROCESS_CASE_SEQ)};
  return obj.at(preprocess_case);
}

}  // namespace oneflow
