#include "oneflow/core/record/image_preprocess.h"

namespace oneflow {

void ImagePreprocessImpl<PreprocessCase::kResize>::DoPreprocess(
    cv::Mat* image, const ImagePreprocess& preprocess_conf) const {
  TODO();
}

void ImagePreprocessImpl<PreprocessCase::kCrop>::DoPreprocess(
    cv::Mat* image, const ImagePreprocess& preprocess_conf) const {
  TODO();
}

ImagePreprocessIf* GetImagePreprocess(PreprocessCase preprocess_case) {
  static const HashMap<int, ImagePreprocessIf*> obj = {

#define MAKE_ENTRY(preprocess_case) \
  {preprocess_case, new ImagePreprocessImpl<preprocess_case>},
      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, PREPROCESS_CASE_SEQ)};
  return obj.at(preprocess_case);
}

}  // namespace oneflow
