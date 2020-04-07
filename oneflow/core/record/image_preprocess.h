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
