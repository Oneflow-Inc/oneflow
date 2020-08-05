/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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

ImagePreprocessIf* GetImagePreprocess(PreprocessCase preprocess_case) {
  static const HashMap<int, ImagePreprocessIf*> obj = {

#define MAKE_ENTRY(preprocess_case) {preprocess_case, new ImagePreprocessImpl<preprocess_case>},
      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, PREPROCESS_CASE_SEQ)};
  return obj.at(preprocess_case);
}

}  // namespace oneflow
