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
    if (crop.has_range()) {
      int32_t full_hrange = (image->rows - height) / 2;
      int32_t cur_hrange = full_hrange;
      if (crop.range().has_height() && crop.range().height() <= full_hrange) {
        CHECK_GE(crop.range().height(), 0);
        cur_hrange = crop.range().height();
      }
      int32_t full_wrange = (image->cols - width) / 2;
      int32_t cur_wrange = full_wrange;
      if (crop.range().has_width() && crop.range().width() <= full_wrange) {
        CHECK_GE(crop.range().width(), 0);
        cur_wrange = crop.range().width();
      }
      x = NextRandomInt() % (cur_wrange * 2 + 1) - cur_wrange + full_wrange;
      y = NextRandomInt() % (cur_hrange * 2 + 1) - cur_hrange + full_hrange;
    } else {
      int32_t x_max = (image->cols - width);
      int32_t y_max = (image->rows - height);
      x = x_max > 0 ? (NextRandomInt() % x_max) : x_max;
      y = y_max > 0 ? (NextRandomInt() % y_max) : y_max;
    }
  } else {
    CHECK_LE(x, image->cols - width);
    CHECK_LE(y, image->rows - height);
  }
  *image = (*image)(cv::Rect(x, y, width, height));
}

void ImagePreprocessImpl<PreprocessCase::kCenterCrop>::DoPreprocess(
    cv::Mat* image, const ImagePreprocess& preprocess_conf,
    std::function<int32_t(void)> NextRandomInt) const {
  CHECK(preprocess_conf.has_center_crop());
  const ImageCenterCrop& conf = preprocess_conf.center_crop();
  int32_t width = conf.width();
  int32_t height = conf.height();
  int32_t middle_width = -1;
  int32_t middle_height = -1;
  float crop_aspect_ratio = width * 1.0 / height;
  CHECK_GT(crop_aspect_ratio, 0);
  if ((image->cols * 1.0 / image->rows) >= crop_aspect_ratio) {
    middle_height = image->rows;
    middle_width = static_cast<int32_t>(middle_height * crop_aspect_ratio);
  } else {
    middle_width = image->cols;
    middle_height = static_cast<int32_t>(middle_width / crop_aspect_ratio);
  }
  CHECK_GT(middle_width, 0);
  CHECK_GT(middle_height, 0);
  int32_t x = (image->cols - middle_width) / 2;
  int32_t y = (image->rows - middle_height) / 2;
  CHECK_GE(x, 0);
  CHECK_GE(y, 0);
  *image = (*image)(cv::Rect(x, y, middle_width, middle_height));
  cv::Mat dst;
  cv::resize(*image, dst, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
  *image = dst;
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

void ImagePreprocessImpl<PreprocessCase::kBgr2Rgb>::DoPreprocess(
    cv::Mat* image, const ImagePreprocess& preprocess_conf,
    std::function<int32_t(void)> NextRandomInt) const {
  CHECK(preprocess_conf.has_bgr2rgb());
  cv::cvtColor(*image, *image, cv::COLOR_BGR2RGB);
}

void ImagePreprocessImpl<PreprocessCase::kCutout>::DoPreprocess(
    cv::Mat* image, const ImagePreprocess& preprocess_conf,
    std::function<int32_t(void)> NextRandomInt) const {
  CHECK(preprocess_conf.has_cutout());
  const ImageCutout& conf = preprocess_conf.cutout();
  const float cutout_ratio = conf.cutout_ratio();
  const int cutout_size = conf.cutout_size();
  const int cutout_mode = conf.cutout_mode();
  const int cutout_filler = conf.cutout_filler();

  if (cutout_ratio > 0.0) {
    if (NextRandomInt() % 100 < cutout_ratio * 100) {
      int m_cutout_size = cutout_size;
      if (cutout_mode == 1) {  // uniform
        m_cutout_size = (NextRandomInt() % 100) * cutout_size / 100;
      }
      int h_off = NextRandomInt() % (image->cols - m_cutout_size - 1);
      int w_off = NextRandomInt() % (image->rows - m_cutout_size - 1);
      for (int h = h_off; h < h_off + m_cutout_size; ++h) {
        for (int w = w_off; w < w_off + m_cutout_size; ++w) {
          *(image->data + image->step[0] * h + image->step[1] * w + image->elemSize1() * 0) =
              (cutout_filler == 0 ? 0 : NextRandomInt() % cutout_filler / 100.);
          *(image->data + image->step[0] * h + image->step[1] * w + image->elemSize1() * 1) =
              (cutout_filler == 0 ? 0 : NextRandomInt() % cutout_filler / 100.);
          *(image->data + image->step[0] * h + image->step[1] * w + image->elemSize1() * 2) =
              (cutout_filler == 0 ? 0 : NextRandomInt() % cutout_filler / 100.);
        }
      }
    }
  }
}

ImagePreprocessIf* GetImagePreprocess(PreprocessCase preprocess_case) {
  static const HashMap<int, ImagePreprocessIf*> obj = {

#define MAKE_ENTRY(preprocess_case) {preprocess_case, new ImagePreprocessImpl<preprocess_case>},
      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, PREPROCESS_CASE_SEQ)};
  return obj.at(preprocess_case);
}

}  // namespace oneflow
