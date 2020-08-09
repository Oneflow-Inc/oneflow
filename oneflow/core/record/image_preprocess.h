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
#ifndef ONEFLOW_CORE_RECORD_IMAGE_PREPROCESS_H_
#define ONEFLOW_CORE_RECORD_IMAGE_PREPROCESS_H_

#include <opencv2/opencv.hpp>
#include "oneflow/core/record/image.pb.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

using PreprocessCase = ImagePreprocess::PreprocessCase;

class ImagePreprocessIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ImagePreprocessIf);
  virtual ~ImagePreprocessIf() = default;

  virtual void DoPreprocess(cv::Mat* image, const ImagePreprocess& preprocess_conf,
                            std::function<int32_t(void)> NextRandomInt) const = 0;

 protected:
  ImagePreprocessIf() = default;

 private:
};

template<PreprocessCase preprocess_case>
class ImagePreprocessImpl;

template<>
class ImagePreprocessImpl<PreprocessCase::kResize> final : public ImagePreprocessIf {
 public:
 private:
  void DoPreprocess(cv::Mat* image, const ImagePreprocess& preprocess_conf,
                    std::function<int32_t(void)> NextRandomInt) const override;
};

template<>
class ImagePreprocessImpl<PreprocessCase::kCrop> final : public ImagePreprocessIf {
 public:
 private:
  void DoPreprocess(cv::Mat* image, const ImagePreprocess& preprocess_conf,
                    std::function<int32_t(void)> NextRandomInt) const override;
};

template<>
class ImagePreprocessImpl<PreprocessCase::kCenterCrop> final : public ImagePreprocessIf {
 public:
 private:
  void DoPreprocess(cv::Mat* image, const ImagePreprocess& preprocess_conf,
                    std::function<int32_t(void)> NextRandomInt) const override;
};

template<>
class ImagePreprocessImpl<PreprocessCase::kCropWithRandomSize> final : public ImagePreprocessIf {
 public:
 private:
  void DoPreprocess(cv::Mat* image, const ImagePreprocess& preprocess_conf,
                    std::function<int32_t(void)> NextRandomInt) const override;
};

template<>
class ImagePreprocessImpl<PreprocessCase::kMirror> final : public ImagePreprocessIf {
 public:
 private:
  void DoPreprocess(cv::Mat* image, const ImagePreprocess& preprocess_conf,
                    std::function<int32_t(void)> NextRandomInt) const override;
};

template<>
class ImagePreprocessImpl<PreprocessCase::kBgr2Rgb> final : public ImagePreprocessIf {
 public:
 private:
  void DoPreprocess(cv::Mat* image, const ImagePreprocess& preprocess_conf,
                    std::function<int32_t(void)> NextRandomInt) const override;
};

#define PREPROCESS_CASE_SEQ                                 \
  OF_PP_MAKE_TUPLE_SEQ(PreprocessCase::kResize)             \
  OF_PP_MAKE_TUPLE_SEQ(PreprocessCase::kMirror)             \
  OF_PP_MAKE_TUPLE_SEQ(PreprocessCase::kCrop)               \
  OF_PP_MAKE_TUPLE_SEQ(PreprocessCase::kCenterCrop)         \
  OF_PP_MAKE_TUPLE_SEQ(PreprocessCase::kCropWithRandomSize) \
  OF_PP_MAKE_TUPLE_SEQ(PreprocessCase::kBgr2Rgb)

ImagePreprocessIf* GetImagePreprocess(PreprocessCase);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_IMAGE_PREPROCESS_H_
